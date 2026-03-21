import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.func import functional_call

class DynamicsModel(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes, learning_rate, seed):
        super().__init__() # initialize nn.Module
        self.observation_dim = observation_dim # number of observation features
        self.action_dim = action_dim # number of action inputs
        self.hidden_sizes = hidden_sizes # MLP hidden layer sizes
        self.learning_rate = learning_rate # optimizer learning rate
        self.seed = seed # seed for reproducibility 
        
        # hold a local set of norm stats
        self.mean_obs = None
        self.std_obs = None
        self.mean_act = None
        self.std_act = None
        self.mean_delta = None
        self.std_delta = None
        
        layers = [] # build MLP layers for delta dynamics model
        input_dim = observation_dim + action_dim # concatenate observation and action as model input
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size)) # Fully connected layer
            layers.append(nn.ReLU()) # Nonlinear activation
            input_dim = hidden_size # next layer input size
            
        layers.append(nn.Linear(input_dim, observation_dim)) # output predicted delta 
        self.model = nn.Sequential(*layers) # MLP that predicts normalized delta
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) # optimizer for training model weights

    def predict_next_state(self, observation, action):
        # normalize transitions
        normalized_observation = self._normalize(observation, self.mean_obs, self.std_obs)
        normalized_action = self._normalize(action, self.mean_act, self.std_act)
        # predict normalized delta from inputs [normalized_observation, normalized_action]
        normalized_delta_prediction = self.model(torch.cat([normalized_observation, normalized_action], dim=-1))
        # unnormalize delta
        delta_prediction = normalized_delta_prediction * self.std_delta + self.mean_delta
        # compute next state as obs + predicted delta
        return observation + delta_prediction
    
    def predict_next_state_with_parameters(self, observation, action, parameters):
        # assert parameters are provided
        if parameters is None: 
            raise ValueError("parameters must be provided for predict_next_state_with_parameters")
        
        # normalize transitions
        normalized_observation = self._normalize(observation, self.mean_obs, self.std_obs)
        normalized_action = self._normalize(action, self.mean_act, self.std_act)
        # predict normalized delta using the given parameters instead of self.model parameters
        normalized_delta_prediction = functional_call(self.model, parameters, (torch.cat([normalized_observation, normalized_action], dim=-1),))
        # unnormalize delta
        delta_prediction = normalized_delta_prediction * self.std_delta + self.mean_delta
        # compute next state as obs + predicted delta
        return observation + delta_prediction
    
    def compute_adapted_parameters(self, support_observations, support_actions, support_next_observations, inner_lr):
        # compute delta targets for support set
        support_delta = support_next_observations - support_observations
        # snapshot current model parameters
        base_parameters = OrderedDict(self.model.named_parameters())
        bias_names = [name for name in base_parameters.keys() if name.endswith(".bias")]
        if len(bias_names) == 0:
            raise RuntimeError("No bias parameters found for adaptation.")
        # compute support loss using the provided parameter
        loss = self.compute_loss_with_parameters(support_observations, support_actions, support_delta, base_parameters)
        # compute gradients only for bias parameters (BitFit inner step)
        bias_params = [base_parameters[name] for name in bias_names]
        bias_grads = torch.autograd.grad(loss, tuple(bias_params), create_graph=True)
        bias_grad_map = {name: grad for name, grad in zip(bias_names, bias_grads)}
        # apply one gradient step to bias parameters only
        adapted_parameters = OrderedDict()
        for name, param in base_parameters.items():
            if name in bias_grad_map:
                adapted_parameters[name] = param - inner_lr * bias_grad_map[name]
            else:
                adapted_parameters[name] = param
        return adapted_parameters

    def compute_loss_with_parameters(self, observation, action, delta, parameters):
        # normalize transitions
        normalized_observation = self._normalize(observation, self.mean_obs, self.std_obs)
        normalized_action = self._normalize(action, self.mean_act, self.std_act)
        normalized_delta = self._normalize(delta, self.mean_delta, self.std_delta)
        # predict normalized delta using the given parameters instead of self.model parameters
        normalized_delta_prediction = functional_call(self.model, parameters, (torch.cat([normalized_observation, normalized_action], dim=-1),))
        # compute mean squared error in normalized space
        return torch.mean((normalized_delta_prediction - normalized_delta) ** 2)

    def compute_loss(self, observation, action, delta):
        # normalize transitions
        normalized_observation = self._normalize(observation, self.mean_obs, self.std_obs)
        normalized_action = self._normalize(action, self.mean_act, self.std_act)
        normalized_delta = self._normalize(delta, self.mean_delta, self.std_delta)
        # predict normalized delta from inputs [normalized_observation, normalized_action]
        normalized_delta_prediction = self.model(torch.cat([normalized_observation, normalized_action], dim=-1))
        # compute mean squared error in normalized space
        return torch.mean((normalized_delta_prediction - normalized_delta) ** 2)
    
    def update_normalization_stats(self, mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta):
        eps = 1e-8
        # takes fresh norm stats computed on latest trajectores and stores them localy
        device = next(self.parameters()).device
        self.mean_obs = torch.as_tensor(mean_obs, dtype=torch.float32, device=device)
        self.std_obs = torch.as_tensor(std_obs, dtype=torch.float32, device=device).clamp_min(eps)
        self.mean_act = torch.as_tensor(mean_act, dtype=torch.float32, device=device)
        self.std_act = torch.as_tensor(std_act, dtype=torch.float32, device=device).clamp_min(eps)
        self.mean_delta = torch.as_tensor(mean_delta, dtype=torch.float32, device=device)
        self.std_delta = torch.as_tensor(std_delta, dtype=torch.float32, device=device).clamp_min(eps)

    def _assert_normalization_stats(self):
        if (self.mean_obs is None or self.std_obs is None or self.mean_act is None or self.std_act is None or self.mean_delta is None or self.std_delta is None):
            raise RuntimeError("DynamicsModel normalization stats are not set yet. ")
        
    def get_parameter_dict(self):
        parameter_dict = OrderedDict(self.model.named_parameters())
        return parameter_dict
        
    def _normalize(self, raw_input, mean, std):
        if (mean is None or std is None): raise RuntimeError("DynamicsModel normalization stats are not set yet")
        return (raw_input - mean) / std
