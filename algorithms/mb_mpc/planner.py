import numpy as np
import torch


class RandomShootingPlanner:
    """
    Random Shooting MPC Planner for MB-MPC.

    At each timestep:
      - Sample many random action sequences from the action space
      - Roll them forward using the learned dynamics model
      - Evaluate total predicted reward
      - Execute the first action of the best sequence
    """

    def __init__(self, dynamics_model, action_space,
                 horizon=20, num_candidates=1000,
                 dt=0.05, ctrl_cost_weight=0.1, device="cpu"):
        """
        Args:
            dynamics_model: trained DynamicsModel (predicts Δs)
            action_space: environment action_space (Box)
            horizon: planning horizon (steps to simulate)
            num_candidates: number of random action sequences
            dt: environment step size (HalfCheetah default = 0.05)
            ctrl_cost_weight: coefficient for action penalty
            device: torch device
        """
        self.model = dynamics_model
        self.action_space = action_space
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.dt = dt
        self.ctrl_cost_weight = ctrl_cost_weight
        self.device = torch.device(device)

    def _compute_reward(self, state, next_state, action):
        """
        HalfCheetah-style reward:
            reward = forward_velocity - ctrl_cost
        """
        x_before = state[:, 0]
        x_after = next_state[:, 0]
        x_velocity = (x_after - x_before) / self.dt
        forward_reward = x_velocity

        ctrl_cost = self.ctrl_cost_weight * torch.sum(action ** 2, dim=-1)

        return forward_reward - ctrl_cost

    def plan(self, state):
        """
        Plan the next action using random shooting MPC.

        Args:
            state: np.ndarray (state_dim,) current state

        Returns:
            action: np.ndarray (action_dim,) first action of best sequence
        """
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        action_dim = self.action_space.shape[0]

        # Sample candidate action sequences uniformly
        candidates = np.random.uniform(
            low=self.action_space.low,
            high=self.action_space.high,
            size=(self.num_candidates, self.horizon, action_dim)
        )
        candidates = torch.tensor(candidates, dtype=torch.float32, device=self.device)

        # Roll each sequence forward in the learned model
        total_rewards = torch.zeros(self.num_candidates, device=self.device)
        current_states = state.repeat(self.num_candidates, 1)

        for t in range(self.horizon):
            actions_t = candidates[:, t, :]  # (num_candidates, action_dim)
            next_states = self.model.predict_next_state(current_states, actions_t)
            reward = self._compute_reward(current_states, next_states, actions_t)

            total_rewards += reward
            current_states = next_states

        # Pick best sequence
        best_idx = torch.argmax(total_rewards).item()
        best_action = candidates[best_idx, 0].cpu().numpy()

        return best_action
