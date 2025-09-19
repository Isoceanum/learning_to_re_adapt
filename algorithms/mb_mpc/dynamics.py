import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256], norm_eps: float = 1e-10):
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
        self._norm_eps = float(norm_eps)
        # Cached torch tensors for fast use on the active device
        self._state_mean_t = None
        self._state_std_t = None
        self._action_mean_t = None
        self._action_std_t = None
        self._delta_mean_t = None
        self._delta_std_t = None
        
        
    def refresh_cached_normalization(self):
        """
        Rebuild cached torch tensors for normalization from already-set
        numpy arrays (state_mean/std, action_mean/std, delta_mean/std)
        without recomputing statistics.
        """
        device = next(self.parameters()).device
        dtype = torch.float32
        if self.state_mean is not None:
            self._state_mean_t = torch.as_tensor(self.state_mean, dtype=dtype, device=device)
        if self.state_std is not None:
            self._state_std_t = torch.as_tensor(self.state_std, dtype=dtype, device=device)
        if self.action_mean is not None:
            self._action_mean_t = torch.as_tensor(self.action_mean, dtype=dtype, device=device)
        if self.action_std is not None:
            self._action_std_t = torch.as_tensor(self.action_std, dtype=dtype, device=device)
        if self.delta_mean is not None:
            self._delta_mean_t = torch.as_tensor(self.delta_mean, dtype=dtype, device=device)
        if self.delta_std is not None:
            self._delta_std_t = torch.as_tensor(self.delta_std, dtype=dtype, device=device)

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
        self.state_std = states.std(0) + self._norm_eps

        self.action_mean = actions.mean(0)
        self.action_std = actions.std(0) + self._norm_eps

        self.delta_mean = deltas.mean(0)
        self.delta_std = deltas.std(0) + self._norm_eps

        # Also cache torch tensors on the current device for fast forward()
        device = next(self.parameters()).device
        dtype = torch.float32
        self._state_mean_t = torch.as_tensor(self.state_mean, dtype=dtype, device=device)
        self._state_std_t = torch.as_tensor(self.state_std, dtype=dtype, device=device)
        self._action_mean_t = torch.as_tensor(self.action_mean, dtype=dtype, device=device)
        self._action_std_t = torch.as_tensor(self.action_std, dtype=dtype, device=device)
        self._delta_mean_t = torch.as_tensor(self.delta_mean, dtype=dtype, device=device)
        self._delta_std_t = torch.as_tensor(self.delta_std, dtype=dtype, device=device)
        
        
    def forward(self, state, action):
        # Normalize inputs (ensure numpy stats are treated as torch tensors on the right device)
        if self._state_mean_t is not None:
            # Use cached per-device tensors
            state_mean = self._state_mean_t
            state_std = self._state_std_t
            action_mean = self._action_mean_t
            action_std = self._action_std_t

            state = (state - state_mean) / state_std
            action = (action - action_mean) / action_std

        x = torch.cat([state, action], dim=-1)
        delta_pred = self.model(x)

        # De-normalize outputs
        if self._delta_mean_t is not None:
            delta_mean = self._delta_mean_t
            delta_std = self._delta_std_t
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
        
        Changed for Nagabandi fidelity: loss in normalized delta space
        The original implementation trains the MLP on normalized inputs
        and targets (Δs). We mirror that by computing MSE between the
        predicted normalized Δs and the normalized target Δs.
        Args:
            state: (batch, state_dim)
            action: (batch, action_dim)
            next_state: (batch, state_dim)
        Returns:
            loss (scalar tensor)
        """
        # Compute target delta in raw space first
        true_delta = next_state - state

        # Normalize inputs if stats are available
        # (Keep inference path unchanged; we bypass forward() to avoid
        # de-normalization here.)
        if self._state_mean_t is not None:
            state_n = (state - self._state_mean_t) / self._state_std_t
            action_n = (action - self._action_mean_t) / self._action_std_t
        else:
            state_n, action_n = state, action

        x = torch.cat([state_n, action_n], dim=-1)
        pred_delta_n = self.model(x)  # prediction in normalized Δs space

        # Normalize target delta if stats are available
        if self._delta_mean_t is not None:
            target_delta_n = (true_delta - self._delta_mean_t) / self._delta_std_t
        else:
            target_delta_n = true_delta

        loss = F.mse_loss(pred_delta_n, target_delta_n)
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


class DynamicsModelProb(nn.Module):
    """
    Probabilistic dynamics: predicts mean and log-variance of Δs (state delta).
    Trains with negative log-likelihood on the normalized Δs with bounded
    log-variance for stability.
    """

    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256], logvar_bounds=(-10.0, 0.5), norm_eps: float = 1e-10):
        super().__init__()

        input_dim = state_dim + action_dim
        output_dim = state_dim

        layers = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(last_dim, output_dim)
        self.logvar_head = nn.Linear(last_dim, output_dim)

        # Normalization stats (numpy for serialization)
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        self.delta_mean = None
        self.delta_std = None
        self._norm_eps = float(norm_eps)
        # Cached torch tensors for fast use on the active device
        self._state_mean_t = None
        self._state_std_t = None
        self._action_mean_t = None
        self._action_std_t = None
        self._delta_mean_t = None
        self._delta_std_t = None

        # Learnable log-variance bounds (PETS-style)
        min_init, max_init = float(logvar_bounds[0]), float(logvar_bounds[1])
        self.min_logvar = nn.Parameter(torch.ones(output_dim) * min_init)
        self.max_logvar = nn.Parameter(torch.ones(output_dim) * max_init)

    def set_logvar_bounds(self, min_val: float, max_val: float):
        """Set learnable logvar bounds (in-place initialization)."""
        with torch.no_grad():
            self.min_logvar.data.fill_(float(min_val))
            self.max_logvar.data.fill_(float(max_val))

    def fit_normalization(self, states, actions, next_states):
        deltas = next_states - states
        self.state_mean = states.mean(0)
        self.state_std = states.std(0) + self._norm_eps
        self.action_mean = actions.mean(0)
        self.action_std = actions.std(0) + self._norm_eps
        self.delta_mean = deltas.mean(0)
        self.delta_std = deltas.std(0) + self._norm_eps
        # Cache torch tensors on current device
        device = next(self.parameters()).device
        dtype = torch.float32
        self._state_mean_t = torch.as_tensor(self.state_mean, dtype=dtype, device=device)
        self._state_std_t = torch.as_tensor(self.state_std, dtype=dtype, device=device)
        self._action_mean_t = torch.as_tensor(self.action_mean, dtype=dtype, device=device)
        self._action_std_t = torch.as_tensor(self.action_std, dtype=dtype, device=device)
        self._delta_mean_t = torch.as_tensor(self.delta_mean, dtype=dtype, device=device)
        self._delta_std_t = torch.as_tensor(self.delta_std, dtype=dtype, device=device)

    def _normalize_inputs(self, state, action):
        if self._state_mean_t is not None:
            state = (state - self._state_mean_t) / self._state_std_t
            action = (action - self._action_mean_t) / self._action_std_t
        return state, action

    def _normalize_delta(self, delta):
        if self._delta_mean_t is not None:
            return (delta - self._delta_mean_t) / self._delta_std_t
        return delta

    def _denormalize_delta(self, delta):
        if self._delta_mean_t is not None:
            return delta * self._delta_std_t + self._delta_mean_t
        return delta

    def forward_norm(self, state, action):
        state, action = self._normalize_inputs(state, action)
        x = torch.cat([state, action], dim=-1)
        h = self.backbone(x)
        mean = self.mean_head(h)
        raw_logvar = self.logvar_head(h)
        # Bound raw_logvar between learnable min/max using softplus barriers
        # max barrier: logvar <= max_logvar
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        # min barrier: logvar >= min_logvar
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def forward(self, state, action):
        mean_n, logvar_n = self.forward_norm(state, action)
        # De-normalize mean
        mean = self._denormalize_delta(mean_n)
        # Adjust log-variance for de-normalization: var_true = var_n * (delta_std^2)
        if self._delta_std_t is not None:
            log_scale = 2.0 * torch.log(self._delta_std_t)
            logvar = logvar_n + log_scale
        else:
            logvar = logvar_n
        return mean, logvar

    def predict_next_state(self, state, action, sample: bool = False):
        mean, logvar = self.forward(state, action)
        if sample:
            std = torch.exp(0.5 * logvar)
            delta = mean + std * torch.randn_like(mean)
        else:
            delta = mean
        return state + delta

    def nll_loss(self, mean_n, logvar_n, target_delta_n):
        # Negative log-likelihood in normalized space + logvar bounds regularization
        inv_var = torch.exp(-logvar_n)
        nll = 0.5 * (logvar_n + (target_delta_n - mean_n) ** 2 * inv_var)
        nll_mean = nll.mean()
        # Regularize learnable bounds to prevent collapse/over-expansion (PETS-style)
        reg_weight = 1e-4
        reg = reg_weight * (torch.sum(self.max_logvar) - torch.sum(self.min_logvar))
        return nll_mean + reg

    def loss_fn(self, state, action, next_state):
        target_delta = next_state - state
        target_delta_n = self._normalize_delta(target_delta)
        mean_n, logvar_n = self.forward_norm(state, action)
        return self.nll_loss(mean_n, logvar_n, target_delta_n)

    def train_step(self, optimizer, state, action, next_state):
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

    def predict_next_state(self, state, action, model_indices=None, sample: bool = False):
        """
        Args:
            state: torch.Tensor (N, state_dim)
            action: torch.Tensor (N, action_dim)
            model_indices: Optional[int or 1D Tensor/ndarray of shape (N,)]
        Returns:
            next_state: torch.Tensor (N, state_dim)
        """
        if self.num_models == 1 and model_indices is None:
            # Try to pass sampling flag if supported
            try:
                return self.models[0].predict_next_state(state, action, sample=sample)
            except TypeError:
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
            # Try to pass sampling flag if supported
            try:
                ns_k = self.models[k].predict_next_state(s_k, a_k, sample=sample)
            except TypeError:
                ns_k = self.models[k].predict_next_state(s_k, a_k)
            next_states[mask] = ns_k

        return next_states
