import numpy as np
import torch


class KStepWindowBuffer:
    """Trajectory buffer that preserves episode order and builds contiguous K-step windows."""

    def __init__(self, valid_split_ratio: float = 0.1, seed: int = 42):
        self.valid_split_ratio = float(valid_split_ratio)
        self.rng = np.random.default_rng(seed)

        self.train_obs = []
        self.train_act = []
        self.train_next_obs = []

        self.val_obs = []
        self.val_act = []
        self.val_next_obs = []

    # --- storage helpers -------------------------------------------------
    def _choose_split(self):
        if len(self.train_obs) == 0:
            return "train"
        if len(self.val_obs) == 0:
            return "val"
        return "val" if self.rng.random() < self.valid_split_ratio else "train"

    def add_trajectory(self, observations, actions, next_observations):
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        next_observations = np.asarray(next_observations, dtype=np.float32)

        if not (len(observations) == len(actions) == len(next_observations)):
            raise ValueError("Observation/action/next_obs length mismatch")

        split = self._choose_split()
        if split == "train":
            self.train_obs.append(observations)
            self.train_act.append(actions)
            self.train_next_obs.append(next_observations)
        else:
            self.val_obs.append(observations)
            self.val_act.append(actions)
            self.val_next_obs.append(next_observations)

    # --- window building --------------------------------------------------
    def _build_windows_from_split(self, split: str, k: int):
        obs_eps = self.train_obs if split == "train" else self.val_obs
        act_eps = self.train_act if split == "train" else self.val_act
        next_obs_eps = self.train_next_obs if split == "train" else self.val_next_obs

        obs_list = []
        act_seq_list = []
        target_seq_list = []

        for obs, act, nxt in zip(obs_eps, act_eps, next_obs_eps):
            T = len(act)
            if T < k:
                continue
            for start in range(0, T - k + 1):
                obs0 = obs[start]
                act_seq = act[start : start + k]
                tgt_seq = nxt[start : start + k]
                obs_list.append(obs0)
                act_seq_list.append(act_seq)
                target_seq_list.append(tgt_seq)

        if len(obs_list) == 0:
            raise RuntimeError(f"No windows of length {k} available for split='{split}'")

        obs_t = torch.as_tensor(np.stack(obs_list, axis=0), dtype=torch.float32)
        act_t = torch.as_tensor(np.stack(act_seq_list, axis=0), dtype=torch.float32)
        tgt_t = torch.as_tensor(np.stack(target_seq_list, axis=0), dtype=torch.float32)
        return obs_t, act_t, tgt_t

    def build_dataloaders(self, k: int, batch_size: int, shuffle: bool = True):
        from torch.utils.data import TensorDataset, DataLoader

        train_obs, train_act, train_tgt = self._build_windows_from_split("train", k)
        val_obs, val_act, val_tgt = self._build_windows_from_split("val", k)

        train_ds = TensorDataset(train_obs, train_act, train_tgt)
        val_ds = TensorDataset(val_obs, val_act, val_tgt)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        stats = {
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
        }
        return train_loader, val_loader, stats

    # --- summary ----------------------------------------------------------
    def num_trajectories(self):
        return {
            "train": len(self.train_obs),
            "val": len(self.val_obs),
        }

