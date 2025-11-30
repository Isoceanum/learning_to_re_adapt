import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call

class DynamicsModel(nn.Module):

    def __init__(self, observation_dim, action_dim, hidden_sizes, learning_rate, train_epochs, valid_split_ratio, patience):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.valid_split_ratio = valid_split_ratio
        self.patience = patience
        
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
        """Predict next state in original (unnormalized) space."""
        self._assert_normalized()
        # --- Normalize inputs to match training ---
        obs_norm = (observation - self.observations_mean) / self.observations_std
        act_norm = (action - self.actions_mean) / self.actions_std

        # --- Predict normalized delta ---
        delta_pred_norm = self.model(torch.cat([obs_norm, act_norm], dim=-1))

        # --- Unnormalize delta back to original scale ---
        delta_pred = delta_pred_norm * self.delta_std + self.delta_mean

        # --- Compute next state prediction in real (unnormalized) space ---
        next_state_pred = observation + delta_pred
        return next_state_pred
    
    
    def predict_next_state_with_parameters(self, observation, action, parameters):
        """Predict next state in original (unnormalized) space using given parameters (θ′)."""
        if parameters is None:
            return self.predict_next_state(observation, action)
        
        self._assert_normalized()
        
         # Normalize inputs to match training
        obs_norm = (observation - self.observations_mean) / self.observations_std
        act_norm = (action - self.actions_mean) / self.actions_std
        
        # Concatenate normalized obs and act, then run the model with θ′
        inputs = torch.cat([obs_norm, act_norm], dim=-1)
        delta_pred_norm = functional_call(self.model, parameters, (inputs,))
        
        # Unnormalize delta and compute next-state prediction
        delta_pred = delta_pred_norm * self.delta_std + self.delta_mean
        next_state_pred = observation + delta_pred
        return next_state_pred
        

    
    def set_normalization_stats(self, stats):
        """Load normalization statistics (means/stds) from a ReplayBuffer."""
        device = next(self.model.parameters()).device    
        self.observations_mean = stats["observations_mean"].to(device)
        self.observations_std = stats["observations_std"].to(device)
        self.actions_mean = stats["actions_mean"].to(device)
        self.actions_std = stats["actions_std"].to(device)
        self.delta_mean = stats["delta_mean"].to(device)
        self.delta_std = stats["delta_std"].to(device)
    
    def update(self, observations, actions, next_observations):
        """Train the dynamics model in normalized space."""

        # Compute loss in normalized space
        loss = self._compute_loss(observations, actions, next_observations)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def fit(self, observations, actions, next_observations):
        # Get the size of the batch 
        batch_size = observations.shape[0]
        
        # Shuffle indices
        shuffled_indices = torch.randperm(batch_size)
        
        # Compute size of validation set based on valid_split_ratio parameter
        validation_set_size = int(batch_size * self.valid_split_ratio)
        
        # Slice indices into train and eval sets
        validation_indices = shuffled_indices[:validation_set_size]
        training_indices = shuffled_indices[validation_set_size:]
        
        # Extract train and eval transitions
        obs_val = observations[validation_indices]
        act_val = actions[validation_indices]
        next_obs_val = next_observations[validation_indices]
        
        obs_train = observations[training_indices]
        act_train = actions[training_indices]
        next_obs_train = next_observations[training_indices]
        
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        last_training_loss = None
        
        # Loop train_epochs times 
        for epoch in range(self.train_epochs):
            # Do one optimization step on  current training batch and return loss
            last_training_loss = self.update(obs_train, act_train, next_obs_train)
            
            with torch.no_grad():
                validation_loss = self._compute_loss(obs_val, act_val, next_obs_val).item()
                
            # If model improved on eval data, we update best_validation_loss and reset epochs_without_improvement
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
            # If model did not improve we increment epochs_without_improvement
            else:
                epochs_without_improvement += 1
                
            # Early stopping if we see no improvment for self.patience epochs
            if epochs_without_improvement >= self.patience:
                break
                
        return {
            "epochs_trained": epoch + 1,
            "train_loss": float(last_training_loss) if last_training_loss is not None else None,
            "val_loss": float(best_validation_loss) if best_validation_loss < float("inf") else None,
        }

    def _compute_loss(self, observations, actions, next_observations):
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
        
        return loss


    
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


