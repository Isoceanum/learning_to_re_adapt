import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.func import functional_call

class DynamicsModel(nn.Module):

    def __init__(self, observation_dim, action_dim, hidden_sizes=[256, 256], learning_rate=1e-3):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        
        layers = []
        input_dim = observation_dim + action_dim # state and action into a single input vector
        
        # Build a simple feedforward MLP to learn f(s, a) → Δs
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size)) # Fully connected layer
            layers.append(nn.ReLU()) # Nonlinear activation
            input_dim = hidden_size
            
        layers.append(nn.Linear(input_dim, observation_dim))
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.observations_mean = None
        self.observations_std = None
        self.actions_mean = None
        self.actions_std = None
        self.delta_mean = None
        self.delta_std = None
        
    def _assert_normalized(self):
        assert all(v is not None for v in ( self.observations_mean, self.observations_std, self.actions_mean, self.actions_std, self.delta_mean, self.delta_std)), "Normalization stats not set on dynamics_model"
    
    def predict_next_state(self, observation, action):
        self._assert_normalized()
            
        # Normalize inputs to match training
        obs_norm = (observation - self.observations_mean) / self.observations_std
        act_norm = (action - self.actions_mean) / self.actions_std

        # Predict normalized delta
        delta_pred_norm = self.model(torch.cat([obs_norm, act_norm], dim=-1))

        # Unnormalize delta back to original scale
        delta_pred = delta_pred_norm * self.delta_std + self.delta_mean

        # Compute next state prediction in real (unnormalized) space
        next_state_pred = observation + delta_pred
        return next_state_pred
    
    def set_normalization_stats(self, stats):
        """Load normalization statistics (means/stds) from a ReplayBuffer."""
        self.observations_mean = stats["observations_mean"]
        self.observations_std = stats["observations_std"]
        self.actions_mean = stats["actions_mean"]
        self.actions_std = stats["actions_std"]
        self.delta_mean = stats["delta_mean"]
        self.delta_std = stats["delta_std"]
    
    def update(self, observations, actions, next_observations):
        """Train the dynamics model in normalized space."""
        
        self._assert_normalized()
        
        #Normalize inputs
        obs_norm = (observations - self.observations_mean) / self.observations_std
        act_norm = (actions - self.actions_mean) / self.actions_std

        # Compute normalized target delta
        target_delta = next_observations - observations
        target_delta_norm = (target_delta - self.delta_mean) / self.delta_std

        # Predict normalized delta
        pred_delta_norm = self.model(torch.cat([obs_norm, act_norm], dim=-1))

        # Compute loss in normalized space
        loss = torch.mean((pred_delta_norm - target_delta_norm) ** 2)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def compute_normalized_delta_loss(self, observations, actions, next_observations):
        self._assert_normalized()
            
        epsilon = 1e-8 # Used to avoid divide by zero or exploding values
        
        # normalized observations and actions
        obs_norm = (observations - self.observations_mean) / (self.observations_std + epsilon)
        act_norm = (actions - self.actions_mean) / (self.actions_std + epsilon)
        
        # Compute unnormalized delta target
        target_delta = next_observations - observations
        
        # Normalize the delta target
        target_delta_norm = (target_delta - self.delta_mean) / (self.delta_std + epsilon)
        
        # Predict the normalized delta using dynamics model
        pred_delta_norm = self.model(torch.cat([obs_norm, act_norm], dim=-1))
        
        # Compute loss in normalized space
        loss = torch.mean((pred_delta_norm - target_delta_norm) ** 2)
        
        return loss
        
    def compute_normalized_delta_loss_with_parameters(self, observations, actions, next_observations, parameters):
        self._assert_normalized()
            
        epsilon = 1e-8 # Used to avoid divide by zero or exploding values
        
        # normalized observations and actions
        obs_norm = (observations - self.observations_mean) / (self.observations_std + epsilon)
        act_norm = (actions - self.actions_mean) / (self.actions_std + epsilon)
        
        # Compute unnormalized delta target
        target_delta = next_observations - observations
        
        # Normalize the delta target
        target_delta_norm = (target_delta - self.delta_mean) / (self.delta_std + epsilon)
        
        # Predict the normalized delta using dynamics model
        pred_delta_norm = functional_call(self.model, parameters, (torch.cat([obs_norm, act_norm], dim=-1),))
        
        # Compute loss in normalized space
        loss = torch.mean((pred_delta_norm - target_delta_norm) ** 2)
        
        return loss
    
