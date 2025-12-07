from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch


class TransitionBuffer:
    """Cumulative, normalized meta-training dataset for GrBAL dynamics."""

    def __init__(self, valid_split_ratio: float = 0.2, seed: Optional[int] = None) -> None:
        """
        Initialize an empty buffer for train/validation splits.

        Args:
            valid_split_ratio: Fraction of incoming trajectories reserved for validation.
            seed: Optional RNG seed for reproducible sampling.
        """
        
        self.valid_split_ratio = valid_split_ratio
        self.seed = seed
        self._generators: dict[tuple[str, Optional[int]], torch.Generator] = {}

        # Storage placeholders (to be initialized lazily).
        self.train_observations = None
        self.train_actions = None
        self.train_delta = None

        self.validation_observations = None
        self.validation_actions = None
        self.validation_delta = None
        
    def add_trajectories(self, observations, actions, deltas) -> None:
        """
        Split the incoming trajectories into train/val and append them to the cumulative dataset.

        Args:
            observations: Shape (num_paths, path_len, obs_dim).
            actions: Shape (num_paths, path_len, act_dim).
            deltas: Shape (num_paths, path_len, obs_dim).
        """
        assert observations.shape[0] == actions.shape[0] == deltas.shape[0]
        
        num_trajectories = observations.shape[0]
        
        num_val = int(num_trajectories * self.valid_split_ratio)
        num_train = num_trajectories - num_val

        generator = self._get_generator(observations.device)
        permutations = torch.randperm(num_trajectories, generator=generator, device=observations.device)

        train_idx = permutations[:num_train]
        val_idx = permutations[num_train:]

        # Slice into train and val batches (still just local, not stored yet)
        train_obs_batch = observations[train_idx]
        train_act_batch = actions[train_idx]
        train_delta_batch = deltas[train_idx]

        val_obs_batch = observations[val_idx]
        val_act_batch = actions[val_idx]
        val_delta_batch = deltas[val_idx]
        
        if self.train_observations is None:
            # First time we see any train data
            self.train_observations = train_obs_batch
            self.train_actions = train_act_batch
            self.train_delta = train_delta_batch
        else:
            # Concatenate new trajectories along the first dimension
            self.train_observations = torch.cat([self.train_observations, train_obs_batch], dim=0)
            self.train_actions = torch.cat([self.train_actions, train_act_batch], dim=0)
            self.train_delta = torch.cat([self.train_delta, train_delta_batch], dim=0)
            
        if self.validation_observations is None:
            self.validation_observations = val_obs_batch
            self.validation_actions = val_act_batch
            self.validation_delta = val_delta_batch
        else:
            self.validation_observations = torch.cat([self.validation_observations, val_obs_batch], dim=0)
            self.validation_actions = torch.cat([self.validation_actions, val_act_batch], dim=0)
            self.validation_delta = torch.cat([self.validation_delta, val_delta_batch], dim=0)


    def num_train_trajectories(self) -> int:
        """Number of training trajectories currently stored."""
        if self.train_observations is None:
            return 0
        return self.train_observations.shape[0]


    def num_validation_trajectories(self) -> int:
        """Number of validation trajectories currently stored."""
        if self.validation_observations is None:
            return 0
        return self.validation_observations.shape[0]

    def train_shape(self) -> tuple[int, int, int] | None:
        """Shape of training observations (num_trajectories, trajectory_len, obs_dim) if present."""
        if self.train_observations is None:
            return None
        return tuple(self.train_observations.shape)

    def val_shape(self) -> tuple[int, int, int] | None:
        """Shape of validation observations (num_trajectories, trajectory_len, obs_dim) if present."""
        if self.validation_observations is None:
            return None
        return tuple(self.validation_observations.shape)
    
    def sample_meta_batch(
        self,
        meta_batch_size: int,
        past_len: int,
        future_len: int,
        split: Literal["train", "val"] = "train",
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Sample a meta-batch of windows from the cumulative dataset.

        Conceptually:
        - Choose `meta_batch_size` random trajectories (paths).
        - For each path, choose a random time index such that there is
          enough room for a window of length (past_len + future_len).
        - Extract that window, then split it into:
            past  = first `past_len` steps
            future = next `future_len` steps

        This method will be used by the GrBAL inner/outer loop:
        - past_* will go to the inner adaptation (θ -> θ′)
        - future_* will go to the outer meta-loss under θ′
        """
        
        if split == "train":
            obs = self.train_observations
            act = self.train_actions
            delta = self.train_delta
        else:
            obs = self.validation_observations
            act = self.validation_actions
            delta = self.validation_delta

        if obs is None:
            raise RuntimeError(f"No data available for split={split} in TransitionBuffer.")
        
        
        # Unpack basic shapes
        num_trajectories, trajectory_len, obs_dim = obs.shape
        _, _, act_dim = act.shape

        window_len = past_len + future_len
        if trajectory_len < window_len:
            raise ValueError(
                f"Trajectory length {trajectory_len} is too short "
                f"for window_len={window_len} (past_len={past_len}, future_len={future_len})."
            )
            
            
        generator = self._get_generator(obs.device)
        traj_indices = torch.randint(
            low=0,
            high=num_trajectories,
            size=(meta_batch_size,),
            device=obs.device,
            generator=generator,
        )
        max_start = trajectory_len - window_len + 1
        start_indices = torch.randint(
            low=0,
            high=max_start,
            size=(meta_batch_size,),
            device=obs.device,
            generator=generator,
        )

        offsets = torch.arange(window_len, device=obs.device)
        time_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)

        sel_obs = obs[traj_indices]
        sel_act = act[traj_indices]
        sel_delta = delta[traj_indices]

        obs_idx = time_indices.unsqueeze(-1).expand(-1, -1, obs_dim)
        act_idx = time_indices.unsqueeze(-1).expand(-1, -1, act_dim)

        windows_obs = torch.gather(sel_obs, dim=1, index=obs_idx)
        windows_act = torch.gather(sel_act, dim=1, index=act_idx)
        windows_delta = torch.gather(sel_delta, dim=1, index=obs_idx)

        past_obs = windows_obs[:, :past_len]
        future_obs = windows_obs[:, past_len:past_len + future_len]

        past_act = windows_act[:, :past_len]
        future_act = windows_act[:, past_len:past_len + future_len]

        past_delta = windows_delta[:, :past_len]
        future_delta = windows_delta[:, past_len:past_len + future_len]

        return past_obs, past_act, past_delta, future_obs, future_act, future_delta



    def is_empty(self) -> bool:
        """Return True if no data has been added yet."""
        return self.train_observations is None and self.validation_observations is None


    def clear(self) -> None:
        """Drop all stored data and reset the buffer to its initial empty state."""
        self.train_observations = None
        self.train_actions = None
        self.train_delta = None

        self.validation_observations = None
        self.validation_actions = None
        self.validation_delta = None

    def _get_generator(self, device) -> torch.Generator:
        torch_device = torch.device(device)
        key = (torch_device.type, torch_device.index)
        if key not in self._generators:
            gen = torch.Generator(device=torch_device)
            if self.seed is not None:
                gen.manual_seed(self.seed)
            self._generators[key] = gen
        return self._generators[key]

# ---------------------------------------------------------------------------
# Must-do notes for this buffer
# - Call add_trajectories with full rollouts shaped (num_paths, path_len, dim);
#   do not feed per-step inserts (remove any ReplayBuffer-style adds).
# - Sample meta windows only via sample_meta_batch; drop segment_sampler usage.
# - Consumers must rebuild next_obs as obs + delta after sampling; this buffer
#   only stores deltas.***
