import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import torch.optim as optim
from torch.func import functional_call

class DynamicsModel(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes, learning_rate, seed):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.seed = seed
        
        # hold a local set of norm stats
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
        loss = self.compute_loss(batch_obs, batch_act, batch_delta)  
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
        
    def predict_next_state(self, observation, action):
        self._assert_normalization_stats()
        obs_norm = (observation - self.mean_obs) / self.std_obs
        act_norm = (action - self.mean_act) / self.std_act
        pred_delta_norm = self.model(torch.cat([obs_norm, act_norm], dim=-1))
        pred_delta = pred_delta_norm * self.std_delta + self.mean_delta
        return observation + pred_delta
    
    def predict_next_state_with_parameters(self, observation, action, parameters):
        if parameters is None: raise ValueError("parameters must be provided for predict_next_state_with_parameters")
        self._assert_normalization_stats()
        obs_norm = (observation - self.mean_obs) / self.std_obs
        act_norm = (action - self.mean_act) / self.std_act
        pred_delta_norm = functional_call(self.model, parameters, (torch.cat([obs_norm, act_norm], dim=-1),))
        pred_delta = pred_delta_norm * self.std_delta + self.mean_delta
        return observation + pred_delta
        
    def compute_adapted_params(self, support_obs, support_act, support_next_obs, inner_lr):
        support_delta = support_next_obs - support_obs
        base_params = OrderedDict(self.model.named_parameters())
        loss = self.compute_loss_with_parameters(support_obs, support_act, support_delta, base_params)
        grads = torch.autograd.grad(loss, tuple(base_params.values()), create_graph=False)
        adapted_params = OrderedDict((name, (param - inner_lr * grad).detach()) for (name, param), grad in zip(base_params.items(), grads))
        return adapted_params

    def compute_loss_with_parameters(self, observation, action, delta, parameters):
        self._assert_normalization_stats()
        obs_norm = (observation - self.mean_obs) / self.std_obs
        act_norm = (action - self.mean_act) / self.std_act
        delta_norm = (delta - self.mean_delta) / self.std_delta
        pred_delta_norm = functional_call(self.model, parameters, (torch.cat([obs_norm, act_norm], dim=-1),))
        return torch.mean((pred_delta_norm - delta_norm) ** 2)

    def compute_loss(self, observation, action, delta):
        self._assert_normalization_stats()
        obs_norm = (observation - self.mean_obs) / self.std_obs
        act_norm = (action - self.mean_act) / self.std_act
        delta_norm = (delta - self.mean_delta) / self.std_delta
        pred_delta_norm = self.model(torch.cat([obs_norm, act_norm], dim=-1))
        return torch.mean((pred_delta_norm - delta_norm) ** 2)
    
    def update_normalization_stats(self, mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta):
        # takes fresh norm stats computed on latest trajectores and stores them localy
        device = next(self.parameters()).device
        self.mean_obs = torch.as_tensor(mean_obs, dtype=torch.float32, device=device)
        self.std_obs = torch.as_tensor(std_obs, dtype=torch.float32, device=device)
        self.mean_act = torch.as_tensor(mean_act, dtype=torch.float32, device=device)
        self.std_act = torch.as_tensor(std_act, dtype=torch.float32, device=device)
        self.mean_delta = torch.as_tensor(mean_delta, dtype=torch.float32, device=device)
        self.std_delta = torch.as_tensor(std_delta, dtype=torch.float32, device=device)

    def _assert_normalization_stats(self):
        if (self.mean_obs is None or self.std_obs is None or self.mean_act is None or self.std_act is None or self.mean_delta is None or self.std_delta is None):
            raise RuntimeError("DynamicsModel normalization stats are not set yet. ")
        
    def get_parameter_dict(self):
        parameter_dict = OrderedDict(self.model.named_parameters())
        return parameter_dict
        
        
        