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
            r_t = self.reward_fn(current_states, next_states, a_t_rep)
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
