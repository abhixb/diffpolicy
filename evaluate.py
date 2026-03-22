"""Evaluation: rollout diffusion policy in ManiSkill environment."""

import argparse
import json
import logging
from collections import deque
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from data.normalize import Normalizer
from model.diffusion import DiffusionPolicy
from model.obs_encoder import StateMLP
from model.unet1d import TemporalUnet

log = logging.getLogger(__name__)


def build_policy(cfg, normalizer: Normalizer, device: torch.device) -> DiffusionPolicy:
    """Reconstruct policy from config."""
    # Infer dims from normalizer
    obs_dim = len(normalizer.stats["obs"]["min"])
    action_dim = len(normalizer.stats["action"]["min"])

    obs_encoder = StateMLP(
        obs_dim=obs_dim,
        obs_horizon=cfg.data.obs_horizon,
        hidden_dim=cfg.model.obs_encoder_hidden,
        num_layers=cfg.model.obs_encoder_layers,
    )
    noise_net = TemporalUnet(
        action_dim=action_dim,
        obs_emb_dim=obs_encoder.output_dim,
        down_dims=list(cfg.model.down_dims),
        diffusion_step_embed_dim=cfg.model.diffusion_step_embed_dim,
        kernel_size=cfg.model.kernel_size,
        n_groups=cfg.model.n_groups,
    )
    policy = DiffusionPolicy(
        noise_net=noise_net,
        obs_encoder=obs_encoder,
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        num_inference_steps=cfg.diffusion.num_inference_steps,
        beta_schedule=cfg.diffusion.beta_schedule,
        prediction_type=cfg.diffusion.prediction_type,
    )
    return policy.to(device)


def flatten_obs(obs) -> np.ndarray:
    """Flatten ManiSkill observation into a 1D vector.

    Handles both flat array (state mode) and dict (structured mode).
    """
    if isinstance(obs, np.ndarray):
        return obs.flatten().astype(np.float32)
    # Torch tensor (ManiSkill3 returns tensors with batch dim)
    if hasattr(obs, 'numpy'):
        return obs.squeeze(0).cpu().numpy().flatten().astype(np.float32)
    # Dict format
    parts = []
    for key in ["agent_qpos", "agent_qvel"]:
        if key in obs:
            parts.append(np.asarray(obs[key]).flatten())
    if "extra" in obs:
        for k in sorted(obs["extra"].keys()):
            parts.append(np.asarray(obs["extra"][k]).flatten())
    return np.concatenate(parts).astype(np.float32)


def evaluate(
    checkpoint_path: str,
    config_path: str,
    normalizer_path: str,
    env_id: str = "PickCube-v1",
    num_episodes: int = 100,
    use_ema: bool = True,
    render: bool = False,
    save_video: bool = False,
    video_dir: str = "eval_videos",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and normalizer
    cfg = OmegaConf.load(config_path)
    normalizer = Normalizer.from_state_dict(json.load(open(normalizer_path)))

    # Build and load policy
    policy = build_policy(cfg, normalizer, device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["ema"] if use_ema and "ema" in ckpt else ckpt["model"]
    policy.load_state_dict(state_dict)
    policy.eval()

    pred_horizon = cfg.data.pred_horizon
    obs_horizon = cfg.data.obs_horizon
    action_exec_horizon = cfg.data.action_exec_horizon
    action_dim = len(normalizer.stats["action"]["min"])
    action_shape = (pred_horizon, action_dim)

    # Create environment
    import mani_skill.envs  # noqa: register envs
    import gymnasium as gym

    render_mode = None
    if render:
        render_mode = "human"
    elif save_video:
        render_mode = "rgb_array"

    control_mode = cfg.eval.get("control_mode", "pd_joint_pos") if hasattr(cfg, "eval") else "pd_joint_pos"
    max_episode_steps = cfg.eval.get("max_episode_steps", 200) if hasattr(cfg, "eval") else 200

    env = gym.make(
        env_id,
        obs_mode="state",
        control_mode=control_mode,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
    )

    if save_video:
        from mani_skill.utils.wrappers import RecordEpisode
        video_path = Path(video_dir)
        video_path.mkdir(parents=True, exist_ok=True)
        env = RecordEpisode(env, output_dir=str(video_path), save_trajectory=False)

    successes = 0
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        obs_history = deque(maxlen=obs_horizon)
        done = False
        step = 0
        exec_idx = action_exec_horizon  # force replan on first step

        while not done:
            flat_obs = flatten_obs(obs)
            obs_t = torch.from_numpy(flat_obs).float().to(device)
            obs_t = normalizer.normalize(obs_t.unsqueeze(0), "obs").squeeze(0)
            obs_history.append(obs_t)

            # Pad history if needed
            while len(obs_history) < obs_horizon:
                obs_history.appendleft(obs_history[0])

            if exec_idx >= action_exec_horizon:
                # Replan
                obs_chunk = torch.stack(list(obs_history), dim=0).unsqueeze(0)  # (1, T_o, obs_dim)
                action_chunk = policy.generate_actions(obs_chunk, action_shape)  # (1, T_a, act_dim)
                action_chunk = normalizer.unnormalize(action_chunk.squeeze(0).cpu(), "action")
                exec_idx = 0

            action = action_chunk[exec_idx].numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            exec_idx += 1
            step += 1

        success = info.get("success", False)
        if isinstance(success, (np.ndarray, torch.Tensor)):
            success = bool(success.item())
        successes += int(success)
        episode_lengths.append(step)

        if (ep + 1) % 10 == 0:
            log.info(f"Episode {ep+1}/{num_episodes}: success_rate={successes/(ep+1):.2%}")

    env.close()
    success_rate = successes / num_episodes
    avg_length = np.mean(episode_lengths)
    log.info(f"Final: success_rate={success_rate:.2%}, avg_episode_length={avg_length:.1f}")
    return {"success_rate": success_rate, "avg_episode_length": avg_length}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", required=True, help="Path to train.yaml config")
    parser.add_argument("--normalizer", required=True, help="Path to normalizer.json")
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save-video", action="store_true", help="Save episode videos")
    parser.add_argument("--video-dir", default="eval_videos", help="Directory to save videos")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        normalizer_path=args.normalizer,
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        use_ema=not args.no_ema,
        render=args.render,
        save_video=args.save_video,
        video_dir=args.video_dir,
    )
