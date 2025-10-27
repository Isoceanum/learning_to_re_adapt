import numpy as np
import torch
from typing import Callable, Optional

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
        discount: float = 1.0,
        reward_fn: Optional[Callable] = None,
        rng: Optional[torch.Generator] = None,  # Added for Nagabandi fidelity
    ):
        self.model = dynamics_model
        self.action_space = action_space
        self.horizon = int(horizon)
        self.n_candidates = int(n_candidates)
        self.device = torch.device(device)
        self.reward_fn = reward_fn
        self.discount = float(discount)
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
            r_t = self.reward_fn(current_states, a_t_rep, next_states)
            if not torch.is_tensor(r_t):
                r_t = torch.as_tensor(r_t, dtype=torch.float32, device=self.device)
            if self.discount == 1.0:
                total_returns += r_t.view(B, N)
            else:
                total_returns += (self.discount ** t) * r_t.view(B, N)
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


class MPPIPlanner:
    """Model Predictive Path Integral (MPPI) planner.

    Implements the sampling-based control update from
    Williams et al., 2015 using soft-importance re-weighting.
    """

    def __init__(
        self,
        dynamics_model,
        horizon: int,
        n_candidates: int,
        temperature: float,
        action_dim: int,
        action_bounds,
        device: str = "cpu",
        reward_fn: Optional[Callable] = None,
        rng: Optional[torch.Generator] = None,
        noise_std=None,
    ):
        if temperature <= 0:
            raise ValueError("MPPI temperature must be positive")

        self.model = dynamics_model
        self.horizon = int(horizon)
        self.n_candidates = int(n_candidates)
        self.lambda_ = float(temperature)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.reward_fn = reward_fn
        self.rng = rng

        low, high = action_bounds
        self._low = torch.as_tensor(low, dtype=torch.float32, device=self.device)
        self._high = torch.as_tensor(high, dtype=torch.float32, device=self.device)

        span = (self._high - self._low).abs().clamp_min(1e-6)
        if noise_std is None:
            base_std = 0.5 * span
        else:
            std = torch.as_tensor(noise_std, dtype=torch.float32, device=self.device)
            if std.ndim == 0:
                if std.item() <= 0:
                    raise ValueError("MPPI noise_std must be positive")
                base_std = torch.full_like(self._high, std.item())
            else:
                if std.shape[-1] != self.action_dim:
                    raise ValueError(
                        f"noise_std must have length {self.action_dim}, got {tuple(std.shape)}"
                    )
                base_std = std.view(-1)
                if torch.any(base_std <= 0):
                    raise ValueError("All entries in noise_std must be positive")

        self._base_std = base_std
        self._mean: Optional[torch.Tensor] = None

        if self.reward_fn is None:
            raise RuntimeError("MPPIPlanner requires a reward_fn")

    def reset(self):
        """Reset the internal trajectory mean buffers."""
        if self._mean is not None:
            self._mean.zero_()

    def set_rng(self, rng: Optional[torch.Generator]):
        self.rng = rng

    def _ensure_buffers(self, batch_size: int) -> None:
        if self._mean is None or self._mean.shape[0] != batch_size:
            self._mean = torch.zeros(batch_size, self.horizon, self.action_dim, device=self.device)

    def _sample_action_sequences(self, batch_size: int) -> torch.Tensor:
        self._ensure_buffers(batch_size)
        shape = (batch_size, self.n_candidates, self.horizon, self.action_dim)
        noise = torch.randn(shape, device=self.device, generator=self.rng)
        mean = self._mean.unsqueeze(1)
        std = self._base_std.view(1, 1, 1, self.action_dim)
        samples = mean + noise * std
        low = self._low.view(1, 1, 1, self.action_dim)
        high = self._high.view(1, 1, 1, self.action_dim)
        samples = torch.minimum(torch.maximum(samples, low), high)
        return samples

    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> np.ndarray:
        state_np = np.asarray(state)
        if state_np.ndim == 1:
            state_np = state_np[None, :]
        elif state_np.ndim != 2:
            raise ValueError(f"state must be (S,) or (B,S), got shape={state_np.shape}")

        state_t = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
        batch = int(state_t.shape[0])

        candidates = self._sample_action_sequences(batch)

        current_states = state_t.unsqueeze(1).expand(batch, self.n_candidates, -1).reshape(batch * self.n_candidates, -1)
        costs = torch.zeros(batch, self.n_candidates, device=self.device)

        for t in range(self.horizon):
            a_t = candidates[:, :, t, :]
            a_flat = a_t.reshape(batch * self.n_candidates, self.action_dim)
            next_states = self.model.predict_next_state(current_states, a_flat)
            rewards = self.reward_fn(current_states, a_flat, next_states)
            if not torch.is_tensor(rewards):
                rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            costs += (-rewards).view(batch, self.n_candidates)
            current_states = next_states

        # Stabilise exponentials by subtracting the minimal cost per batch element.
        costs_shifted = costs - costs.min(dim=1, keepdim=True).values
        weights = torch.exp(-costs_shifted / self.lambda_)
        weights_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / weights_sum

        weighted_sequences = torch.einsum("bn,bnha->bha", weights, candidates)
        first_actions = weighted_sequences[:, 0, :]

        # Update internal mean with a receding-horizon shift.
        self._ensure_buffers(batch)
        new_mean = torch.zeros_like(self._mean)
        if self.horizon > 1:
            new_mean[:, :-1, :] = weighted_sequences[:, 1:, :]
        self._mean = new_mean

        out = first_actions.detach().cpu().numpy()
        return out[0] if out.shape[0] == 1 else out

    @torch.no_grad()
    def plan(self, state: np.ndarray) -> np.ndarray:
        return self.get_action(state)
