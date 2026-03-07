import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import torch.optim as optim
from pathlib import Path

class DynamicsModel(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes, learning_rate, seed):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.seed = seed
        
        self.mean_obs = None
        self.std_obs = None
        self.mean_act = None
        self.std_act = None
        self.mean_delta = None
        self.std_delta = None
        
        layers = []
        input_dim = observation_dim + action_dim # state and action into a single input vector
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size)) # Fully connected layer
            layers.append(nn.ReLU()) # Nonlinear activation
            input_dim = hidden_size
            
        layers.append(nn.Linear(input_dim, observation_dim))
        self.model = nn.Sequential(*layers)    
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def train_on_batch(self, batch_obs, batch_act, batch_next_obs):
        batch_delta = batch_next_obs - batch_obs
        loss = self.loss(batch_obs, batch_act, batch_delta)  
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
        
    def _forward(self, observation, action):
        x = torch.cat([observation, action], dim=-1)
        delta = self.model(x)
        return delta
        
    def predict_next_state(self, observation, action):
        self._assert_normalization_stats()
        obs_norm = (observation - self.mean_obs) / self.std_obs
        act_norm = (action - self.mean_act) / self.std_act
        pred_delta_norm = self._forward(obs_norm, act_norm)
        pred_delta = pred_delta_norm * self.std_delta + self.mean_delta
        return observation + pred_delta
    
    def loss(self, observation, action, delta):
        self._assert_normalization_stats()
        obs_norm = (observation - self.mean_obs) / self.std_obs
        act_norm = (action - self.mean_act) / self.std_act
        delta_norm = (delta - self.mean_delta) / self.std_delta
        pred_delta_norm = self._forward(obs_norm, act_norm)
        return torch.mean((pred_delta_norm - delta_norm) ** 2)
            
    def update_normalization_stats(self, norm_stats):
        device = next(self.parameters()).device
        self.mean_obs = torch.as_tensor(norm_stats["mean_obs"], dtype=torch.float32, device=device)
        self.std_obs = torch.as_tensor(norm_stats["std_obs"], dtype=torch.float32, device=device)
        self.mean_act = torch.as_tensor(norm_stats["mean_act"], dtype=torch.float32, device=device)
        self.std_act = torch.as_tensor(norm_stats["std_act"], dtype=torch.float32, device=device)
        self.mean_delta = torch.as_tensor(norm_stats["mean_delta"], dtype=torch.float32, device=device)
        self.std_delta = torch.as_tensor(norm_stats["std_delta"], dtype=torch.float32, device=device)
        self._assert_normalization_stats()

    def _assert_normalization_stats(self):
        if (self.mean_obs is None or self.std_obs is None or self.mean_act is None or self.std_act is None or self.mean_delta is None or self.std_delta is None):
            raise RuntimeError("DynamicsModel normalization stats are not set yet.")
        
    def load_saved_model(self, model_path):
        device = next(self.parameters()).device
        payload = torch.load(model_path, map_location=device)
        
        if "state_dict" not in payload: raise KeyError(f"Checkpoint missing 'state_dict' in {model_path}")
        if "norm_stats" not in payload: raise KeyError(f"Checkpoint missing 'norm_stats' in {model_path}")

        self.load_state_dict(payload["state_dict"], strict=True)
        self.update_normalization_stats(payload["norm_stats"])

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

            
        