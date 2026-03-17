"""1-D Temporal U-Net for noise prediction in diffusion policy."""

import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32) * -emb)
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_ch),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResBlock(nn.Module):
    """Residual block with FiLM conditioning from diffusion timestep."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        self.conv1 = Conv1dBlock(in_ch, out_ch, kernel_size, n_groups)
        self.conv2 = Conv1dBlock(out_ch, out_ch, kernel_size, n_groups)
        # FiLM: project cond → (scale, shift)
        self.cond_proj = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch * 2),
        )
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
            cond: (B, cond_dim)
        """
        h = self.conv1(x)
        # FiLM conditioning
        scale_shift = self.cond_proj(cond).unsqueeze(-1)  # (B, 2*C, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.conv2(h)
        return h + self.residual(x)


class TemporalUnet(nn.Module):
    """1D U-Net for action sequence denoising.

    Takes noisy actions (B, T_a, action_dim), observation embedding (B, obs_emb_dim),
    and diffusion timestep (B,), and predicts noise (B, T_a, action_dim).
    """

    def __init__(
        self,
        action_dim: int,
        obs_emb_dim: int,
        down_dims: list[int] = [128, 256, 512],
        diffusion_step_embed_dim: int = 128,
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        # Diffusion timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim

        # Project obs embedding into the conditioning space
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_emb_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Input projection: action_dim → down_dims[0]
        in_ch = action_dim
        all_dims = [in_ch] + list(down_dims)

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(down_dims)):
            self.down_blocks.append(
                ConditionalResBlock(all_dims[i], all_dims[i + 1], cond_dim, kernel_size, n_groups)
            )
            self.down_samples.append(nn.Conv1d(all_dims[i + 1], all_dims[i + 1], 3, stride=2, padding=1))

        # Bottleneck
        self.mid_block = ConditionalResBlock(down_dims[-1], down_dims[-1], cond_dim, kernel_size, n_groups)

        # Decoder (upsampling path)
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(down_dims))):
            self.up_samples.append(nn.ConvTranspose1d(all_dims[i + 1], all_dims[i + 1], 4, stride=2, padding=1))
            # Skip connection doubles the channels
            self.up_blocks.append(
                ConditionalResBlock(all_dims[i + 1] * 2, all_dims[i], cond_dim, kernel_size, n_groups)
            )

        # Output projection
        self.out_proj = nn.Conv1d(all_dims[0], action_dim, 1)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        obs_emb: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_actions: (B, T_a, action_dim)
            obs_emb: (B, obs_emb_dim)
            timestep: (B,) integer diffusion timesteps
        Returns:
            predicted noise: (B, T_a, action_dim)
        """
        # (B, T, C) → (B, C, T) for conv1d
        x = noisy_actions.permute(0, 2, 1)

        # Conditioning: timestep + obs
        t_emb = self.time_emb(timestep)  # (B, cond_dim)
        o_emb = self.obs_proj(obs_emb)   # (B, cond_dim)
        cond = t_emb + o_emb

        # Encoder
        skips = []
        for down_block, down_sample in zip(self.down_blocks, self.down_samples):
            x = down_block(x, cond)
            skips.append(x)
            x = down_sample(x)

        # Bottleneck
        x = self.mid_block(x, cond)

        # Decoder
        for up_sample, up_block, skip in zip(self.up_samples, self.up_blocks, reversed(skips)):
            x = up_sample(x)
            # Handle size mismatch from stride-2 down/up
            if x.shape[-1] != skip.shape[-1]:
                x = x[..., : skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = up_block(x, cond)

        x = self.out_proj(x)
        return x.permute(0, 2, 1)  # (B, T_a, action_dim)
