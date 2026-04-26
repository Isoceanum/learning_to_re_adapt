import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.func import functional_call
from algorithms.meta_learned_low_rank_adaptation.lora_linear import LoRALinear


class DynamicsModel(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes, learning_rate, lora_rank, lora_alpha, seed):
        super().__init__() # initialize nn.Module
        self.observation_dim = observation_dim # number of observation features
        self.action_dim = action_dim # number of action inputs
        self.hidden_sizes = hidden_sizes # MLP hidden layer sizes
        self.learning_rate = learning_rate # optimizer learning rate
        self.seed = seed # seed for reproducibility 
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
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

            layers.append(LoRALinear(input_dim, hidden_size, self.lora_rank, self.lora_alpha))
            layers.append(nn.ReLU()) # Nonlinear activation
            input_dim = hidden_size # next layer input size
            
        layers.append(LoRALinear(input_dim, observation_dim, self.lora_rank, self.lora_alpha))
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
    
    def compute_adapted_parameters(self, support_observations, support_actions, support_next_observations, inner_lr, create_graph):
        # compute delta targets for support set
        support_delta = support_next_observations - support_observations
        # snapshot current model parameters
        base_parameters = OrderedDict(self.model.named_parameters())
        lora_parameters = OrderedDict((name, param) for name, param in base_parameters.items() if name.endswith(".A") or name.endswith(".B"))
        if not lora_parameters: 
            raise RuntimeError("No LoRA parameters found during call compute_adapted_parameters")
        
        # compute support loss using the provided parameter
        loss = self.compute_loss_with_parameters(support_observations, support_actions, support_delta, base_parameters)
        # compute gradients d(loss)/d(params) for the support loss
        gradients = torch.autograd.grad(loss, tuple(lora_parameters.values()), create_graph=create_graph)
        adapted_lora_parameters = OrderedDict((name, param - inner_lr * grad) for (name, param), grad in zip(lora_parameters.items(), gradients))
        adapted_parameters = OrderedDict(base_parameters)
        adapted_parameters.update(adapted_lora_parameters)
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

    def merge_lora_parameters_for_planning(self, parameters):
        merged_parameters = OrderedDict(parameters)
        for module_name, module in self.model.named_modules():
            if not isinstance(module, LoRALinear) or module.r <= 0:
                continue

            prefix = f"{module_name}."
            weight_key = f"{prefix}base.weight"
            a_key = f"{prefix}A"
            b_key = f"{prefix}B"

            delta_weight = (merged_parameters[b_key] @ merged_parameters[a_key]) * module.scaling
            merged_parameters[weight_key] = merged_parameters[weight_key] + delta_weight
            merged_parameters[a_key] = torch.zeros_like(merged_parameters[a_key])
            merged_parameters[b_key] = torch.zeros_like(merged_parameters[b_key])

        return merged_parameters

    def set_lora_merged_flag(self, merged: bool):
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.merged = bool(merged)
        
    def _normalize(self, raw_input, mean, std):
        if (mean is None or std is None): raise RuntimeError("DynamicsModel normalization stats are not set yet")
        return (raw_input - mean) / std
