import numpy as np
import torch


class CEMPlanner:
    """
    Cross-Entropy Method (CEM) MPC Planner for MB-MPC.

    At each timestep, iteratively optimizes a Gaussian distribution over action
    sequences by sampling, evaluating with the learned dynamics, and refitting
    to top-performing elites.

    Interface: call plan(state) to get an action for the current step.
    """

    def __init__(
        self,
        dynamics_model,
        action_space,
        horizon: int = 20,
        num_candidates: int = 1000,
        device: str = "cpu",
        dt: float = 0.05,
        ctrl_cost_weight: float = 0.1,
        num_elites: int = 100,
        max_iters: int = 5,
        alpha: float = 0.1,
    ):
        """
        Args:
            dynamics_model: trained DynamicsModel (predicts Δs)
            action_space: environment action_space (Box)
            horizon: planning horizon (steps to simulate)
            num_candidates: number of candidate action sequences per CEM iteration
            device: torch device string
            dt: environment step size (HalfCheetah default = 0.05)
            ctrl_cost_weight: coefficient for action penalty
            num_elites: how many top sequences to use for refitting
            max_iters: number of CEM iterations
            alpha: smoothing factor for mean/std updates
        """
        self.model = dynamics_model
        self.action_space = action_space
        self.horizon = int(horizon)
        self.num_candidates = int(num_candidates)
        self.dt = float(dt)
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        self.num_elites = int(num_elites)
        self.max_iters = int(max_iters)
        self.alpha = float(alpha)

        self.device = torch.device(device)

        # Bounds as torch tensors
        self._low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
        self._high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)

    def _compute_reward(self, state, next_state, action):
        """
        Hopper-style reward:
          forward_reward (x-velocity) + healthy_reward - ctrl_cost.
        Assumes observations include x-position at index 0 so that
        x-velocity can be computed via finite difference.
        Shapes:
          state/next_state: (N, state_dim), action: (N, action_dim)
        """
        x_before = state[:, 0]
        x_after = next_state[:, 0]
        x_velocity = (x_after - x_before) / self.dt
        forward_reward = x_velocity  # forward_reward_weight = 1.0 by default

        # Healthy reward is a per-step constant in Hopper when
        # terminate_when_unhealthy=True (default). Mirror that here.
        healthy_reward = torch.ones_like(x_velocity)

        ctrl_cost = self.ctrl_cost_weight * torch.sum(action ** 2, dim=-1)
        return forward_reward + healthy_reward - ctrl_cost

    def plan(self, state):
        """
        Plan the next action using CEM.

        Args:
            state: np.ndarray (state_dim,) current state

        Returns:
            action: np.ndarray (action_dim,) first action of the optimized mean sequence
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_dim = int(self.action_space.shape[0])

        # Initialize Gaussian over sequences: mean=0, std=1 (then clamped to bounds when sampling)
        mean = torch.zeros(self.horizon, action_dim, device=self.device)
        std = torch.ones(self.horizon, action_dim, device=self.device)

        for itr in range(self.max_iters):
            # Sample candidates: (num_candidates, horizon, action_dim)
            eps = torch.randn(self.num_candidates, self.horizon, action_dim, device=self.device)
            candidates = mean.unsqueeze(0) + std.unsqueeze(0) * eps

            # Clamp to action bounds
            candidates = torch.max(torch.min(candidates, self._high), self._low)

            with torch.no_grad():
                # Rollout using dynamics model
                total_rewards = torch.zeros(self.num_candidates, device=self.device)
                current_states = state.repeat(self.num_candidates, 1)

                for t in range(self.horizon):
                    actions_t = candidates[:, t, :]  # (num_candidates, action_dim)
                    next_states = self.model.predict_next_state(current_states, actions_t)
                    reward = self._compute_reward(current_states, next_states, actions_t)
                    total_rewards += reward
                    current_states = next_states

            # Select elites
            elite_vals, elite_idx = torch.topk(total_rewards, k=self.num_elites, largest=True, sorted=False)
            elites = candidates[elite_idx]  # (num_elites, horizon, action_dim)

            # Refit Gaussian to elites
            new_mean = elites.mean(dim=0)
            new_std = elites.std(dim=0, unbiased=False).clamp_min(1e-6)

            # Smoothed update
            mean = self.alpha * new_mean + (1.0 - self.alpha) * mean
            std = self.alpha * new_std + (1.0 - self.alpha) * std

        # Take first action from the final mean sequence and clamp
        first_action = mean[0]
        first_action = torch.max(torch.min(first_action, self._high), self._low)
        return first_action.detach().cpu().numpy()

 
