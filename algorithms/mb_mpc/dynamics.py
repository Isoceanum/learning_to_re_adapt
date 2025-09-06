import torch
import torch.nn as nn
import torch.nn.functional as F



class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super().__init__()
        
        input_dim = state_dim + action_dim
        output_dim = state_dim   # predict Δs, same dimension as state

        # Build hidden layers
        layers = []
        last_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(nn.ReLU())   # ReLU activations
            last_dim = hidden_size

        # Output layer (linear, no activation)
        layers.append(nn.Linear(last_dim, output_dim))

        # Wrap into Sequential
        self.model = nn.Sequential(*layers)
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        self.delta_mean = None
        self.delta_std = None
        
        
    def fit_normalization(self, states, actions, next_states):
        """
        Compute mean/std for states, actions, and deltas from a dataset.
        Args:
            states: np.ndarray [N, state_dim]
            actions: np.ndarray [N, action_dim]
            next_states: np.ndarray [N, state_dim]
        """
        deltas = next_states - states

        self.state_mean = states.mean(0)
        self.state_std = states.std(0) + 1e-8

        self.action_mean = actions.mean(0)
        self.action_std = actions.std(0) + 1e-8

        self.delta_mean = deltas.mean(0)
        self.delta_std = deltas.std(0) + 1e-8
        
        
    def forward(self, state, action):
        # Normalize inputs (ensure numpy stats are treated as torch tensors on the right device)
        if self.state_mean is not None:
            state_mean = torch.as_tensor(self.state_mean, dtype=state.dtype, device=state.device)
            state_std = torch.as_tensor(self.state_std, dtype=state.dtype, device=state.device)
            action_mean = torch.as_tensor(self.action_mean, dtype=action.dtype, device=action.device)
            action_std = torch.as_tensor(self.action_std, dtype=action.dtype, device=action.device)

            state = (state - state_mean) / state_std
            action = (action - action_mean) / action_std

        x = torch.cat([state, action], dim=-1)
        delta_pred = self.model(x)

        # De-normalize outputs
        if self.delta_mean is not None:
            delta_mean = torch.as_tensor(self.delta_mean, dtype=delta_pred.dtype, device=delta_pred.device)
            delta_std = torch.as_tensor(self.delta_std, dtype=delta_pred.dtype, device=delta_pred.device)
            delta_pred = delta_pred * delta_std + delta_mean

        return delta_pred

    def predict_next_state(self, state, action):
        """
        Convenience method: predict next state = s + Δs.
        """
        delta_pred = self.forward(state, action)
        return state + delta_pred
    
    
    def loss_fn(self, state, action, next_state):
        """
        Compute MSE loss between predicted Δs and true Δs.
        Args:
            state: (batch, state_dim)
            action: (batch, action_dim)
            next_state: (batch, state_dim)
        Returns:
            loss (scalar tensor)
        """
        true_delta = next_state - state
        pred_delta = self.forward(state, action)
        loss = F.mse_loss(pred_delta, true_delta)
        return loss

    def train_step(self, optimizer, state, action, next_state):
        """
        Perform one gradient update step.
        """
        loss = self.loss_fn(state, action, next_state)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


class EnsembleDynamics:
    """
    Lightweight wrapper around a list of DynamicsModel instances to provide
    a unified predict_next_state interface that supports trajectory sampling (TS1).

    If `model_indices` is provided as a 1D tensor/array of length N, each sample
    in the batch uses the corresponding model. If it is an int, the entire batch
    uses that model. If None, uses the first model (or the only model).
    """

    def __init__(self, models):
        assert isinstance(models, (list, tuple)) and len(models) >= 1
        self.models = list(models)

    @property
    def num_models(self):
        return len(self.models)

    def predict_next_state(self, state, action, model_indices=None):
        """
        Args:
            state: torch.Tensor (N, state_dim)
            action: torch.Tensor (N, action_dim)
            model_indices: Optional[int or 1D Tensor/ndarray of shape (N,)]
        Returns:
            next_state: torch.Tensor (N, state_dim)
        """
        if self.num_models == 1 and model_indices is None:
            return self.models[0].predict_next_state(state, action)

        import torch

        N = state.shape[0]
        if model_indices is None:
            # Default to model 0
            model_indices = torch.zeros(N, dtype=torch.long, device=state.device)
        elif isinstance(model_indices, int):
            model_indices = torch.full((N,), int(model_indices), dtype=torch.long, device=state.device)
        else:
            model_indices = torch.as_tensor(model_indices, dtype=torch.long, device=state.device)
            if model_indices.ndim == 0:
                model_indices = model_indices.view(1).repeat(N)

        next_states = torch.empty_like(state)
        # Route each subset to the corresponding model
        for k in torch.unique(model_indices).tolist():
            k = int(k)
            mask = (model_indices == k)
            if not torch.any(mask):
                continue
            s_k = state[mask]
            a_k = action[mask]
            ns_k = self.models[k].predict_next_state(s_k, a_k)
            next_states[mask] = ns_k

        return next_states
