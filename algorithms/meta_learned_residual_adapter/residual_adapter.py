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
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size

        self.last = nn.Linear(input_dim, observation_dim)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)
        layers.append(self.last)
    
        self.model = nn.Sequential(*layers)

    def forward(self, observation_norm, action_norm, base_pred_next_observation_norm):
        x = torch.cat([observation_norm, action_norm, base_pred_next_observation_norm], dim=-1)
        correction_norm = self.model(x)
        return correction_norm
