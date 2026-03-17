"""Per-dimension min/max normalization to [-1, 1]."""

import torch
import numpy as np


class Normalizer:
    """Normalizes data to [-1, 1] using per-dimension min/max statistics."""

    def __init__(self, stats: dict[str, np.ndarray] | None = None):
        self.stats = stats  # {"obs": {"min": ..., "max": ...}, "action": {...}}

    @staticmethod
    def compute_stats(data: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "min": data.min(axis=0).astype(np.float32),
            "max": data.max(axis=0).astype(np.float32),
        }

    def normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        s = self.stats[key]
        mn = torch.as_tensor(s["min"], dtype=x.dtype, device=x.device)
        mx = torch.as_tensor(s["max"], dtype=x.dtype, device=x.device)
        rang = mx - mn
        rang = rang.clamp(min=1e-6)
        return 2.0 * (x - mn) / rang - 1.0

    def unnormalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        s = self.stats[key]
        mn = torch.as_tensor(s["min"], dtype=x.dtype, device=x.device)
        mx = torch.as_tensor(s["max"], dtype=x.dtype, device=x.device)
        rang = mx - mn
        rang = rang.clamp(min=1e-6)
        return (x + 1.0) / 2.0 * rang + mn

    def state_dict(self) -> dict:
        return {
            k: {kk: vv.tolist() for kk, vv in v.items()}
            for k, v in self.stats.items()
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "Normalizer":
        stats = {
            k: {kk: np.array(vv, dtype=np.float32) for kk, vv in v.items()}
            for k, v in d.items()
        }
        return cls(stats)
