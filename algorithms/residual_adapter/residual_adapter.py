import torch.nn as nn
import torch

class ResidualAdapter(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes, learning_rate, seed):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.seed = seed
        self.learning_rate = learning_rate
        
        layers = []
        input_dim = observation_dim + action_dim 
        
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
        self.mean_delta = None
        self.std_delta = None
        
    def forward(self, observation, action):
        self._assert_normalization_stats()
        obs_norm = (observation - self.mean_obs) / self.std_obs
        act_norm = (action - self.mean_act) / self.std_act

        x = torch.cat([obs_norm, act_norm], dim=-1)
        delta_correction_norm = self.model(x)
        return delta_correction_norm
        
    def update_normalization_stats(self, mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta):
        device = next(self.parameters()).device
        self.mean_obs = torch.as_tensor(mean_obs, dtype=torch.float32, device=device)
        self.std_obs = torch.as_tensor(std_obs, dtype=torch.float32, device=device)
        self.mean_act = torch.as_tensor(mean_act, dtype=torch.float32, device=device)
        self.std_act = torch.as_tensor(std_act, dtype=torch.float32, device=device)
        self.mean_delta = torch.as_tensor(mean_delta, dtype=torch.float32, device=device)
        self.std_delta = torch.as_tensor(std_delta, dtype=torch.float32, device=device)

    def _assert_normalization_stats(self):
        if (self.mean_obs is None or self.std_obs is None or
            self.mean_act is None or self.std_act is None or
            self.mean_delta is None or self.std_delta is None):
            raise RuntimeError("ResidualHead normalization stats are not set yet.")
