import numpy as np
import torch
from typing import Callable, Optional


class NagabandiCEMPlanner:
    """
    CEM MPC planner with semantics matching the classic Nagabandi code:
      - Deterministic dynamics rollouts (no sampling)
      - Returns accumulate over horizon with discount=1
      - Elite refit uses percent_elites and sets std to elites' std (no smoothing)
      - Mean smoothing: mean = alpha * mean + (1 - alpha) * elites_mean
      - Actions sampled from N(mean, std) then clipped to action bounds

    Shapes follow the original: candidates (n, h, a), flatted sequences for CEM stats.

    Expects a torch-compatible `reward_fn(state, next_state, action) -> (N,)`.
    """

    def __init__(
        self,
        dynamics_model,
        action_space,
        horizon: int = 10,
        n_candidates: int = 1024,
        num_cem_iters: int = 8,
        percent_elites: float = 0.1,
        alpha: float = 0.1,
        device: str = "cpu",
        reward_fn: Optional[Callable] = None,
        clip_rollouts: bool = False,
        rng: Optional[torch.Generator] = None,  # Added for Nagabandi fidelity
    ):
        self.model = dynamics_model
        self.action_space = action_space
        self.horizon = int(horizon)
        self.n_candidates = int(n_candidates)
        self.num_cem_iters = int(num_cem_iters)
        self.percent_elites = float(percent_elites)
        self.alpha = float(alpha)
        self.device = torch.device(device)
        self.reward_fn = reward_fn
        self.clip_rollouts = bool(clip_rollouts)
        # Added for Nagabandi fidelity: seeded RNG
        self.rng = rng

        # Bounds as torch tensors
        self._low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
        self._high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)

        # Planner re-initializes its sampling distribution each step

    @torch.no_grad()
    def plan(self, state: np.ndarray) -> np.ndarray:
        """
        Plan action(s) for the given state(s) using CEM.

        Args:
            state: np.ndarray of shape (state_dim,) or (batch, state_dim)
        Returns:
            action: np.ndarray of shape (action_dim,) if input was 1D,
                    else (batch, action_dim)
        """
        # Normalize input to (B, S)
        state_np = np.asarray(state)
        if state_np.ndim == 1:
            state_np = state_np[None, :]
        elif state_np.ndim != 2:
            raise ValueError(f"state must be (S,) or (B,S), got shape={state_np.shape}")
        state_t = torch.tensor(state_np, dtype=torch.float32, device=self.device)  # (B, S)

        B = int(state_t.shape[0])
        action_dim = int(self.action_space.shape[0])
        H = self.horizon
        N = self.n_candidates

        # Initialize mean/std over sequences (per step)
        mean = torch.zeros(B, H, action_dim, device=self.device)
        # Following original code: start with std of ones in action units
        std = torch.ones_like(mean)  # (B, H, A)

        # Precompute repeated bounds for clipping
        low = self._low
        high = self._high

        # Build flattened view helpers
        flat_dim = H * action_dim

        last_returns = None  # (B, N)
        for _ in range(self.num_cem_iters):
            # Sample candidates N(H, A) from N(mean, std)
            # Added for Nagabandi fidelity: use provided generator if set
            eps = torch.randn(B, N, H, action_dim, device=self.device, generator=self.rng)
            candidates_unc = mean.unsqueeze(1) + std.unsqueeze(1) * eps  # (B, N, H, A)
            # Clipped view for stats
            candidates_clip = torch.max(torch.min(candidates_unc, high), low)
            # Which to use for rollouts
            candidates = candidates_clip if self.clip_rollouts else candidates_unc  # (B, N, H, A)

            # Rollout deterministically under the dynamics
            current_states = state_t.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)  # (B*N, S)
            total_returns = torch.zeros(B, N, device=self.device)

            for t in range(H):
                a_t = candidates[:, :, t, :]  # (B, N, A)
                a_t_rep = a_t.reshape(B * N, -1)  # (B*N, A)
                next_states = self.model.predict_next_state(current_states, a_t_rep)
                if self.reward_fn is None:
                    raise RuntimeError("NagabandiCEMPlanner requires a reward_fn")
                r_t = self.reward_fn(current_states, next_states, a_t_rep)
                if not torch.is_tensor(r_t):
                    r_t = torch.as_tensor(r_t, dtype=torch.float32, device=self.device)
                total_returns += r_t.view(B, N)
                current_states = next_states

            # Keep for final action selection
            last_returns = total_returns

            # Select elites per original semantics
            # Flatten sequences for stats: (N, H*A)
            cand_flat = candidates_clip.view(B, N, flat_dim)  # (B, N, H*A)
            # Top-k indices by return
            k = max(1, int(self.percent_elites * N))
            _, elite_idx = torch.topk(total_returns, k=k, dim=1, largest=True, sorted=False)  # (B, k)
            # Build gather indices for each batch
            batch_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, k)  # (B, k)
            elites = cand_flat[batch_idx, elite_idx, :]  # (B, k, H*A)
            # Aggregate elites across batch (matches original broadcast update)
            elites_flat = elites.reshape(B * k, flat_dim)
            elites_mean = elites_flat.mean(dim=0)  # (H*A)
            elites_std = elites_flat.std(dim=0, unbiased=False)  # (H*A)

            # Smooth mean per original: mean = alpha*mean + (1-alpha)*elites_mean
            mean = self.alpha * mean.view(B, flat_dim) + (1.0 - self.alpha) * elites_mean  # broadcast over B
            mean = mean.view(B, H, action_dim)
            # Replace std with elites' std (no smoothing)
            std = elites_std.view(1, H, action_dim).expand(B, -1, -1).clamp_min(1e-6)

        # No cross-step caching

        # Select per-batch best candidate from final iteration
        assert last_returns is not None
        best_idx = torch.argmax(last_returns, dim=1)  # (B,)
        first_actions_all = candidates[:, :, 0, :]  # (B, N, A)
        first_actions = first_actions_all[torch.arange(B, device=self.device), best_idx, :]  # (B, A)
        # Match original Nagabandi semantics: do not clip the selected first action here
        out = first_actions.detach().cpu().numpy()
        # If input was 1D, return 1D action
        return out[0] if out.shape[0] == 1 else out


class RandomShootingPlanner:
    """
    Random Shooting MPC planner matching the original Nagabandi default
    when `use_cem=False`:
      - Uniformly samples N action sequences within action bounds
      - Deterministic dynamics rollouts over horizon
      - Returns with discount=1 and selects the best sequence's first action

    """

    def __init__(
        self,
        dynamics_model,
        action_space,
        horizon: int = 20,
        n_candidates: int = 2000,
        device: str = "cpu",
        reward_fn: Optional[Callable] = None,
        rng: Optional[torch.Generator] = None,  # Added for Nagabandi fidelity
    ):
        self.model = dynamics_model
        self.action_space = action_space
        self.horizon = int(horizon)
        self.n_candidates = int(n_candidates)
        self.device = torch.device(device)
        self.reward_fn = reward_fn
        # Added for Nagabandi fidelity: seeded RNG
        self.rng = rng

        self._low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
        self._high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)

    def _uniform_actions(self, shape):
        """Sample uniformly in [low, high] with optional seeded generator."""
        low = self._low
        high = self._high
        # torch.rand in [0,1), scale to [low, high]
        u = torch.rand(shape, device=self.device, generator=self.rng)
        return low + u * (high - low)

    @torch.no_grad()
    def plan(self, state: np.ndarray) -> np.ndarray:
        state_np = np.asarray(state)
        if state_np.ndim == 1:
            state_np = state_np[None, :]
        elif state_np.ndim != 2:
            raise ValueError(f"state must be (S,) or (B,S), got shape={state_np.shape}")
        state_t = torch.tensor(state_np, dtype=torch.float32, device=self.device)

        B = int(state_t.shape[0])
        A = int(self.action_space.shape[0])
        H = self.horizon
        N = self.n_candidates

        # Sample N action sequences uniformly per batch
        candidates = self._uniform_actions((B, N, H, A))  # (B, N, H, A)

        # Rollout deterministically
        current_states = state_t.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        total_returns = torch.zeros(B, N, device=self.device)

        if self.reward_fn is None:
            raise RuntimeError("RandomShootingPlanner requires a reward_fn")

        for t in range(H):
            a_t = candidates[:, :, t, :]
            a_t_rep = a_t.reshape(B * N, -1)
            next_states = self.model.predict_next_state(current_states, a_t_rep)
            r_t = self.reward_fn(current_states, next_states, a_t_rep)
            if not torch.is_tensor(r_t):
                r_t = torch.as_tensor(r_t, dtype=torch.float32, device=self.device)
            total_returns += r_t.view(B, N)
            current_states = next_states

        # Select best
        best_idx = torch.argmax(total_returns, dim=1)  # (B,)
        first_actions = candidates[:, :, 0, :][torch.arange(B, device=self.device), best_idx, :]
        out = first_actions.detach().cpu().numpy()
        return out[0] if out.shape[0] == 1 else out

    # Added for Nagabandi fidelity: allow setting/updating RNG
    def set_rng(self, rng: Optional[torch.Generator]):
        self.rng = rng

    # Keep API parity with CEM planner
    def reset(self):
        pass

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, value: int):
        self._horizon = int(value)
