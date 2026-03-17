"""Observation encoders: state MLP and (optionally) image CNN."""

import torch
import torch.nn as nn


class StateMLP(nn.Module):
    """Simple MLP that encodes a flattened observation history into a fixed-size embedding."""

    def __init__(self, obs_dim: int, obs_horizon: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        input_dim = obs_dim * obs_horizon
        layers = []
        in_d = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_d, hidden_dim), nn.Mish()])
            in_d = hidden_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, T_o, obs_dim)
        Returns:
            (B, output_dim)
        """
        B = obs.shape[0]
        return self.net(obs.reshape(B, -1))
