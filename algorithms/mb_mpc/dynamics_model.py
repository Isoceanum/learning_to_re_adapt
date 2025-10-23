import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn


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
    
    def predict_next_state(self, observation, action):
        x = torch.cat([observation, action], dim=-1) # merge observation and action into single vector
        delta_pred = self.model(x) # pass input to MLP
        next_state_pred = observation + delta_pred # apply delta to get next state
        return next_state_pred
    
    def update(self, observations, actions, next_observations):
        target_delta = next_observations - observations
        pred_delta = self.model(torch.cat([observations, actions], dim=-1))
        loss = torch.mean((pred_delta - target_delta) ** 2)
        
        self.optimizer.zero_grad() # clear gradients
        loss.backward() # compute new gradient
        self.optimizer.step() # update weights using optimizer
        return loss.item()

    
    