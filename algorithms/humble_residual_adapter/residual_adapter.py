import torch.nn as nn
import torch

class ResidualAdapter(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        layers = []
        input_dim = observation_dim + action_dim + observation_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size)) # Fully connected layer
            layers.append(nn.ReLU()) # Nonlinear activation
            input_dim = hidden_size
            
        layers.append(nn.Linear(input_dim, observation_dim))
        self.model = nn.Sequential(*layers)
        
        self.mean_obs = None
        self.std_obs = None
        
        self.mean_act = None
        self.std_act = None
        
        self.base_delta_mean = None
        self.base_delta_std = None
        
        self.residual_mean = None
        self.residual_std = None
        
    def forward(self, observation, action, base_pred_delta):
        self._assert_normalization_stats()

        obs_norm = (observation - self.mean_obs) / self.std_obs
        act_norm = (action - self.mean_act) / self.std_act
        base_pred_delta_norm = (base_pred_delta - self.base_delta_mean) / self.base_delta_std

        x = torch.cat([obs_norm, act_norm, base_pred_delta_norm], dim=-1)
        delta_correction_norm = self.model(x)
        return delta_correction_norm
        
    def update_normalization_stats(self, mean_obs, std_obs, mean_act, std_act, mean_base_delta, std_base_delta, mean_residual, std_residual):
        device = next(self.parameters()).device

        self.mean_obs = torch.as_tensor(mean_obs, dtype=torch.float32, device=device)
        self.std_obs = torch.as_tensor(std_obs, dtype=torch.float32, device=device)

        self.mean_act = torch.as_tensor(mean_act, dtype=torch.float32, device=device)
        self.std_act = torch.as_tensor(std_act, dtype=torch.float32, device=device)

        self.base_delta_mean = torch.as_tensor(mean_base_delta, dtype=torch.float32, device=device)
        self.base_delta_std = torch.as_tensor(std_base_delta, dtype=torch.float32, device=device)

        self.residual_mean = torch.as_tensor(mean_residual, dtype=torch.float32, device=device)
        self.residual_std = torch.as_tensor(std_residual, dtype=torch.float32, device=device)

    def _assert_normalization_stats(self):
        if (self.mean_obs is None or self.std_obs is None or
            self.mean_act is None or self.std_act is None or
            self.base_delta_mean is None or self.base_delta_std is None or
            self.residual_mean is None or self.residual_std is None):
            raise RuntimeError("ResidualAdapter normalization stats are not set yet.")