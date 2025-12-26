import torch


class ResidualDynamicsWrapper:
    def __init__(self, base_dynamics_model, residual_adapter=None):
        self.base = base_dynamics_model
        self.residual_adapter = residual_adapter

    def set_residual_adapter(self, residual_adapter):
        self.residual_adapter = residual_adapter

    def predict_next_state(self, observation, action):
        if self.residual_adapter is None: 
            return self.base.predict_next_state(observation, action)
        
        base_delta_norm = self.base.predict_state_delta_norm(observation, action)
        residual_delta_norm = self.residual_adapter(observation, action)
        delta_norm = base_delta_norm + residual_delta_norm
        delta = delta_norm * self.base.std_delta + self.base.mean_delta
        return observation + delta

    def loss(self, obs, act, next_obs):
        if self.residual_adapter is None:
            raise RuntimeError("No residual adapter found.")
        
        pred_next = self.predict_next_state(obs, act)
        return torch.mean((pred_next - next_obs) ** 2)
