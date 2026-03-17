"""HDF5 dataset with observation/action chunking for diffusion policy."""

import json
import pathlib

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .normalize import Normalizer


class ManiSkillDiffusionDataset(Dataset):
    """Loads ManiSkill HDF5 demonstrations and returns chunked (obs, action) pairs.

    Each sample is:
        obs_chunk:    (T_o, obs_dim)   — observation history
        action_chunk: (T_a, action_dim) — future action targets
    """

    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        normalizer: Normalizer | None = None,
    ):
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon

        # Load all trajectories into memory
        all_obs, all_actions, traj_ends = self._load_hdf5(dataset_path)

        # Compute or reuse normalizer
        if normalizer is None:
            stats = {
                "obs": Normalizer.compute_stats(all_obs),
                "action": Normalizer.compute_stats(all_actions),
            }
            self.normalizer = Normalizer(stats)
        else:
            self.normalizer = normalizer

        self.obs = torch.from_numpy(all_obs).float()
        self.actions = torch.from_numpy(all_actions).float()
        self.traj_ends = set(traj_ends)

        # Build valid indices: each index i means we can form
        #   obs[i - T_o + 1 : i + 1] and action[i : i + T_a]
        # while staying within a single trajectory.
        self.indices = self._build_indices(traj_ends)

    def _load_hdf5(self, path: str):
        """Load observations and actions from ManiSkill HDF5 replay."""
        all_obs = []
        all_actions = []
        traj_ends = []
        global_idx = 0

        with h5py.File(path, "r") as f:
            # ManiSkill3 stores trajectories as traj_0, traj_1, ...
            traj_keys = sorted(
                [k for k in f.keys() if k.startswith("traj_")],
                key=lambda x: int(x.split("_")[1]),
            )
            for key in traj_keys:
                traj = f[key]
                obs_data = traj["obs"]

                # ManiSkill3 state mode: obs is a flat (T, obs_dim) array
                # Older format: obs is an HDF5 group with agent_qpos, etc.
                if isinstance(obs_data, h5py.Dataset):
                    obs = np.array(obs_data).astype(np.float32)
                else:
                    obs_parts = []
                    for obs_key in ["agent_qpos", "agent_qvel"]:
                        if obs_key in obs_data:
                            obs_parts.append(np.array(obs_data[obs_key]))
                    if "extra" in obs_data:
                        for extra_key in sorted(obs_data["extra"].keys()):
                            data = np.array(obs_data["extra"][extra_key])
                            if data.ndim == 1:
                                data = data[:, None]
                            obs_parts.append(data)
                    obs = np.concatenate(obs_parts, axis=-1).astype(np.float32)
                actions = np.array(traj["actions"]).astype(np.float32)

                # obs may have one more step than actions (final obs)
                T = min(len(obs), len(actions))
                all_obs.append(obs[:T])
                all_actions.append(actions[:T])
                global_idx += T
                traj_ends.append(global_idx)

        return (
            np.concatenate(all_obs, axis=0),
            np.concatenate(all_actions, axis=0),
            traj_ends,
        )

    def _build_indices(self, traj_ends: list[int]):
        """Build list of valid start indices for chunking."""
        indices = []
        traj_start = 0
        for traj_end in traj_ends:
            traj_len = traj_end - traj_start
            # Need at least obs_horizon for obs and pred_horizon for actions
            for i in range(traj_start, traj_end):
                obs_start = i - self.obs_horizon + 1
                action_end = i + self.pred_horizon
                # obs start must be within trajectory
                if obs_start < traj_start:
                    continue
                # action end must be within trajectory
                if action_end > traj_end:
                    continue
                indices.append(i)
            traj_start = traj_end
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        obs_start = i - self.obs_horizon + 1
        obs_chunk = self.obs[obs_start : i + 1]  # (T_o, obs_dim)
        action_chunk = self.actions[i : i + self.pred_horizon]  # (T_a, act_dim)

        obs_chunk = self.normalizer.normalize(obs_chunk, "obs")
        action_chunk = self.normalizer.normalize(action_chunk, "action")

        return obs_chunk, action_chunk

    def save_normalizer(self, path: str):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.normalizer.state_dict(), f)

    @staticmethod
    def load_normalizer(path: str) -> Normalizer:
        with open(path) as f:
            return Normalizer.from_state_dict(json.load(f))
