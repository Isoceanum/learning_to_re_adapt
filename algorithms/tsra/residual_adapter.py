import torch.nn as nn
import torch

class ResidualAdapter(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        layers = []
        input_dim = observation_dim + action_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size

        self.last = nn.Linear(input_dim, observation_dim)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)
        layers.append(self.last)
    
        self.model = nn.Sequential(*layers)

    def forward(self, obs_norm, act_norm):
        x = torch.cat([obs_norm, act_norm], dim=-1)
        delta_correction_norm = self.model(x)
        return delta_correction_norm
