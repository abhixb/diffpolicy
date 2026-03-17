"""DDPM/DDIM diffusion utilities wrapping the noise prediction network."""

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


class DiffusionPolicy(nn.Module):
    """Wraps a noise prediction network with DDPM training and DDIM inference."""

    def __init__(
        self,
        noise_net: nn.Module,
        obs_encoder: nn.Module,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
    ):
        super().__init__()
        self.noise_net = noise_net
        self.obs_encoder = obs_encoder

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )
        self.num_inference_steps = num_inference_steps

    def compute_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Training loss: MSE between predicted and true noise.

        Args:
            obs: (B, T_o, obs_dim)
            actions: (B, T_a, action_dim)
        """
        B = actions.shape[0]
        # Encode observations
        obs_emb = self.obs_encoder(obs)

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.train_scheduler.config.num_train_timesteps,
            (B,), device=actions.device, dtype=torch.long,
        )

        # Sample noise and create noisy actions
        noise = torch.randn_like(actions)
        noisy_actions = self.train_scheduler.add_noise(actions, noise, timesteps)

        # Predict noise
        noise_pred = self.noise_net(noisy_actions, obs_emb, timesteps)

        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def generate_actions(self, obs: torch.Tensor, action_shape: tuple[int, int]) -> torch.Tensor:
        """DDIM inference: denoise from pure noise to action sequence.

        Args:
            obs: (B, T_o, obs_dim)
            action_shape: (T_a, action_dim)
        Returns:
            actions: (B, T_a, action_dim) in normalized space
        """
        B = obs.shape[0]
        device = obs.device

        obs_emb = self.obs_encoder(obs)

        # Start from pure noise
        noisy_actions = torch.randn(B, *action_shape, device=device)

        self.inference_scheduler.set_timesteps(self.num_inference_steps, device=device)

        for t in self.inference_scheduler.timesteps:
            timestep = t.expand(B)
            noise_pred = self.noise_net(noisy_actions, obs_emb, timestep)
            noisy_actions = self.inference_scheduler.step(
                noise_pred, t, noisy_actions
            ).prev_sample

        return noisy_actions
