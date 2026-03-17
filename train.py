"""Training entry point for diffusion policy."""

import copy
import json
import logging
import pathlib

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ManiSkillDiffusionDataset
from model.diffusion import DiffusionPolicy
from model.obs_encoder import StateMLP
from model.unet1d import TemporalUnet

log = logging.getLogger(__name__)


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    def update(self, model: torch.nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].lerp_(v, 1 - self.decay)

    def apply(self, model: torch.nn.Module):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.training.seed)

    # --- Dataset ---
    dataset = ManiSkillDiffusionDataset(
        dataset_path=cfg.data.dataset_path,
        pred_horizon=cfg.data.pred_horizon,
        obs_horizon=cfg.data.obs_horizon,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    log.info(f"Dataset: {len(dataset)} samples, obs_dim={dataset.obs.shape[-1]}, action_dim={dataset.actions.shape[-1]}")

    obs_dim = dataset.obs.shape[-1]
    action_dim = dataset.actions.shape[-1]

    # Save normalizer
    out_dir = pathlib.Path(".")
    dataset.save_normalizer(str(out_dir / "normalizer.json"))

    # --- Model ---
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
    ).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    log.info(f"Model parameters: {n_params:,}")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.num_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.use_fp16)
    ema = EMAModel(policy, decay=cfg.training.ema_decay)

    # --- W&B ---
    if cfg.wandb.enabled:
        try:
            import wandb
            wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True))
        except Exception as e:
            log.warning(f"W&B init failed: {e}")
            cfg.wandb.enabled = False

    # --- Training Loop ---
    global_step = 0
    for epoch in range(cfg.training.num_epochs):
        policy.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (obs, actions) in enumerate(tqdm(loader, desc=f"Epoch {epoch}", leave=False)):
            obs = obs.to(device)
            actions = actions.to(device)

            with torch.amp.autocast("cuda", enabled=cfg.training.use_fp16):
                loss = policy.compute_loss(obs, actions) / cfg.training.grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg.training.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(policy)
                global_step += 1

                if cfg.wandb.enabled and global_step % cfg.training.log_every == 0:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item() * cfg.training.grad_accum_steps,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    }, step=global_step)

            epoch_loss += loss.item() * cfg.training.grad_accum_steps
            num_batches += 1

        lr_scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        log.info(f"Epoch {epoch}: loss={avg_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.2e}")

        # Checkpoint
        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            ckpt_path = out_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model": policy.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }, ckpt_path)
            log.info(f"Saved checkpoint: {ckpt_path}")

    # Save final
    torch.save({
        "epoch": cfg.training.num_epochs - 1,
        "global_step": global_step,
        "model": policy.state_dict(),
        "ema": ema.state_dict(),
    }, out_dir / "checkpoint_final.pt")
    log.info("Training complete.")


if __name__ == "__main__":
    main()
