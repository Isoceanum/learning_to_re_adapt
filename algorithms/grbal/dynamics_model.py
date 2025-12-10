import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
from collections import OrderedDict
from algorithms.grbal.transition_buffer import TransitionBuffer
import torch.optim as optim

class DynamicsModel(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes, learning_rate, train_epochs, valid_split_ratio, meta_batch_size, adapt_batch_size, inner_lr, inner_steps, rolling_average_persitency, seed):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.valid_split_ratio = valid_split_ratio
        self.meta_batch_size = meta_batch_size
        self.window_half_length = adapt_batch_size
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.rolling_average_persitency = rolling_average_persitency
        self.seed = seed
        
        self.transition_buffer = TransitionBuffer(valid_split_ratio, self.seed)# update dynamic model to take inn a seed 

        layers = []
        input_dim = observation_dim + action_dim # state and action into a single input vector
        
        # Build a simple feedforward MLP to learn f(s, a) → Δs
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size)) # Fully connected layer
            layers.append(nn.ReLU()) # Nonlinear activation
            input_dim = hidden_size
            
        layers.append(nn.Linear(input_dim, observation_dim))
        self.model = nn.Sequential(*layers)    
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.observations_mean = None
        self.observations_std = None
        
        self.actions_mean = None
        self.actions_std = None
        
        self.delta_mean = None
        self.delta_std = None
        
        self.total_gd_steps = 0
        
    # grbal 
    def _assert_normalized(self):
        assert all(v is not None for v in ( self.observations_mean, self.observations_std, self.actions_mean, self.actions_std, self.delta_mean, self.delta_std)), "Normalization stats not set on dynamics_model"

    # grbal
    def _normalize_data(self, observations, actions, delta):
        epsilon = 1e-8 # Used to avoid divide by zero or exploding values
        self._assert_normalized()
        
        obs_norm = (observations - self.observations_mean) / (self.observations_std + epsilon)
        act_norm = (actions - self.actions_mean) / (self.actions_std + epsilon)
        delta_norm = (delta - self.delta_mean) / (self.delta_std + epsilon)
        return obs_norm, act_norm, delta_norm

    # grbal
    def _compute_and_set_normalization(self, observations, actions, delta):
        epsilon = 1e-8 # Used to avoid divide by zero or exploding values

        self.observations_mean = observations.mean(dim=(0, 1))
        self.observations_std  = observations.std(dim=(0, 1)) + epsilon

        self.actions_mean = actions.mean(dim=(0, 1))
        self.actions_std  = actions.std(dim=(0, 1)) + epsilon

        self.delta_mean = delta.mean(dim=(0, 1))
        self.delta_std  = delta.std(dim=(0, 1)) + epsilon
    
    # TODO OCEAN this should be simplefied into helper functions
    def fit(self, observations, actions, next_observations):
        
        # 1 Check that inputs have the corretc dimeitons
        assert torch.is_tensor(observations), "fit expects torch.Tensors"
        assert torch.is_tensor(actions), "fit expects torch.Tensors"
        assert torch.is_tensor(next_observations), "fit expects torch.Tensors"
        
        assert observations.ndim == 3
        assert actions.ndim == 3
        assert next_observations.ndim == 3
        assert observations.shape[0] == actions.shape[0] == next_observations.shape[0]
        assert observations.shape[1] == actions.shape[1] == next_observations.shape[1]
        
        assert observations.shape[2] == next_observations.shape[2] == self.observation_dim
        assert actions.shape[2] == self.action_dim
        
        # 2 compute normilizations values
        
        deltas = next_observations - observations
        self._compute_and_set_normalization(observations, actions, deltas)
        observations, actions, deltas = self._normalize_data(observations, actions, deltas)
    
        # 1) Add new rollouts to the cumulative dataset
        self.transition_buffer.add_trajectories(observations, actions, deltas)

        # 2) Build train/val tensors from the *entire* cumulative buffer
        train_obs_3d = self.transition_buffer.train_observations      # (N_train_traj, T, obs_dim)
        val_obs_3d = self.transition_buffer.validation_observations   # (N_val_traj, T, obs_dim)

        # Hyperparameters for meta-training (align with TF defaults when possible)
        meta_batch_size = self.meta_batch_size
        batch_size = self.window_half_length  # half-window length (past/query)
        inner_lr = self.inner_lr
        inner_steps = self.inner_steps
        rolling_average_persitency = self.rolling_average_persitency

        # Steps per epoch heuristic (mirrors TF)
        def _num_steps(split_obs):
            if split_obs is None or split_obs.shape[0] == 0:
                return 0
            total_steps = split_obs.shape[0] * split_obs.shape[1]
            return max(int(total_steps / (meta_batch_size * batch_size * 2)), 1)

        num_steps_per_epoch = _num_steps(train_obs_3d)
        num_steps_test = _num_steps(val_obs_3d)

        valid_loss_rolling_average = None
        valid_loss_rolling_average_prev = None
        last_training_loss = None
        last_valid_loss = None
        gd_steps_this_fit = 0

        for epoch in range(self.train_epochs):
            pre_batch_losses = []
            post_batch_losses = []

            # Training meta-steps
            for _ in range(num_steps_per_epoch):
                (
                    past_obs,
                    past_act,
                    past_delta,
                    future_obs,
                    future_act,
                    future_delta,
                ) = self.transition_buffer.sample_meta_batch(
                    meta_batch_size=meta_batch_size,
                    past_len=batch_size,
                    future_len=batch_size,
                    split="train",
                )

                query_losses = []
                support_losses = []

                for i in range(meta_batch_size):
                    s_obs = past_obs[i].reshape(batch_size, self.observation_dim)
                    s_act = past_act[i].reshape(batch_size, self.action_dim)

                    # Inner adaptation (θ -> θ′) for task i
                    parameters = OrderedDict((n, p) for n, p in self.model.named_parameters())
                    for _ in range(inner_steps):
                        inner_loss = self.compute_normalized_delta_loss_with_parameters(
                            s_obs,
                            s_act,
                            target_delta=past_delta[i].reshape(batch_size, self.observation_dim),
                            already_normalized=True,
                            parameters=parameters,
                        )
                        grads = torch.autograd.grad(
                            inner_loss,
                            tuple(parameters.values()),
                            create_graph=True,
                            allow_unused=True,
                        )
                        updated = OrderedDict()
                        for (name, param), grad in zip(parameters.items(), grads):
                            clean_grad = torch.zeros_like(param) if grad is None else grad
                            updated[name] = param - inner_lr * clean_grad
                        parameters = updated

                    support_losses.append(
                        self.compute_normalized_delta_loss(
                            s_obs,
                            s_act,
                            target_delta=past_delta[i].reshape(batch_size, self.observation_dim),
                            already_normalized=True,
                        ).detach()
                    )

                    q_obs = future_obs[i].reshape(batch_size, self.observation_dim)
                    q_act = future_act[i].reshape(batch_size, self.action_dim)

                    q_loss = self.compute_normalized_delta_loss_with_parameters(
                        q_obs,
                        q_act,
                        target_delta=future_delta[i].reshape(batch_size, self.observation_dim),
                        already_normalized=True,
                        parameters=parameters,
                    )
                    query_losses.append(q_loss)

                if not query_losses:
                    continue

                loss_outer = torch.stack(query_losses).mean()
                last_training_loss = loss_outer.detach()
                self.optimizer.zero_grad()
                loss_outer.backward()
                self.optimizer.step()
                gd_steps_this_fit += 1
                for p in self.model.parameters():
                    p.grad = None

                pre_batch_losses.append(
                    torch.stack(support_losses).mean().item() if support_losses else 0.0
                )
                post_batch_losses.append(loss_outer.item())

            # Validation meta-steps
            valid_losses = []
            for _ in range(num_steps_test):
                (
                    past_obs,
                    past_act,
                    past_delta,
                    future_obs,
                    future_act,
                    future_delta,
                ) = self.transition_buffer.sample_meta_batch(
                    meta_batch_size=meta_batch_size,
                    past_len=batch_size,
                    future_len=batch_size,
                    split="val",
                )

                query_losses = []
                for i in range(meta_batch_size):
                    s_obs = past_obs[i].reshape(batch_size, self.observation_dim)
                    s_act = past_act[i].reshape(batch_size, self.action_dim)

                    parameters = OrderedDict((n, p) for n, p in self.model.named_parameters())
                    for _ in range(inner_steps):
                        inner_loss = self.compute_normalized_delta_loss_with_parameters(
                            s_obs,
                            s_act,
                            target_delta=past_delta[i].reshape(batch_size, self.observation_dim),
                            already_normalized=True,
                            parameters=parameters,
                        )
                        grads = torch.autograd.grad(
                            inner_loss,
                            tuple(parameters.values()),
                            create_graph=False,
                            allow_unused=True,
                        )
                        updated = OrderedDict()
                        for (name, param), grad in zip(parameters.items(), grads):
                            clean_grad = torch.zeros_like(param) if grad is None else grad
                            updated[name] = param - inner_lr * clean_grad
                        parameters = updated

                    q_obs = future_obs[i].reshape(batch_size, self.observation_dim)
                    q_act = future_act[i].reshape(batch_size, self.action_dim)

                    q_loss = self.compute_normalized_delta_loss_with_parameters(
                        q_obs,
                        q_act,
                        target_delta=future_delta[i].reshape(batch_size, self.observation_dim),
                        already_normalized=True,
                        parameters=parameters,
                    )
                    query_losses.append(q_loss.detach())

                if query_losses:
                    valid_losses.append(torch.stack(query_losses).mean().item())

            valid_loss = float("inf") if not valid_losses else float(sum(valid_losses) / len(valid_losses))
            last_valid_loss = valid_loss

            if valid_loss_rolling_average is None:
                valid_loss_rolling_average = 1.5 * valid_loss
                valid_loss_rolling_average_prev = valid_loss_rolling_average
            else:
                valid_loss_rolling_average_prev = valid_loss_rolling_average
                valid_loss_rolling_average = rolling_average_persitency * valid_loss_rolling_average + (1.0 - rolling_average_persitency) * valid_loss

            if valid_loss_rolling_average_prev is not None and valid_loss_rolling_average > valid_loss_rolling_average_prev:
                break
            
        self.total_gd_steps += gd_steps_this_fit

        return {
            "epochs_trained": epoch + 1,
            "train_loss": float(last_training_loss) if last_training_loss is not None else None,
            "val_loss": float(valid_loss_rolling_average) if valid_loss_rolling_average is not None else (float(last_valid_loss) if last_valid_loss is not None else None),
            "gradient_descent_steps": gd_steps_this_fit,
        }

    # === GrBAL meta-learning API ===
    def compute_normalized_delta_loss(self, observations, actions, target_delta, already_normalized=False):
        if target_delta is None:
            raise ValueError("target_delta must be provided.")
        if already_normalized:
            obs_norm, act_norm, target_delta_norm = observations, actions, target_delta
        else:
            obs_norm, act_norm, target_delta_norm = self._normalize_data(observations, actions, target_delta)
        
        # flatten  normalized values from (num_trajectories, T, action_dim) to (N, action_dim)
        normalized_observations_flat = obs_norm.reshape(-1, self.observation_dim)
        normalized_actions_flat = act_norm.reshape(-1, self.action_dim)
        normalized_target_delta_flat = target_delta_norm.reshape(-1, self.observation_dim)
        normalized_input_flat = torch.cat([normalized_observations_flat, normalized_actions_flat], dim=-1)
    
        predicted_normalized_delta_flat = self.model(normalized_input_flat)
        loss = torch.mean((predicted_normalized_delta_flat - normalized_target_delta_flat) ** 2)
        return loss

    def compute_normalized_delta_loss_with_parameters(self, observations, actions, target_delta, already_normalized=False, parameters=None):
        
        if parameters is None:
            raise ValueError("parameters must be provided for compute_normalized_delta_loss_with_parameters")
        
        if target_delta is None:
            raise ValueError("target_delta must be provided.")
        
        if already_normalized:
            obs_norm, act_norm, target_delta_norm = observations, actions, target_delta
        else:
            obs_norm, act_norm, target_delta_norm = self._normalize_data(observations, actions, target_delta)
        
        # flatten  normalized values from (num_trajectories, T, action_dim) to (N, action_dim)
        normalized_observations_flat = obs_norm.reshape(-1, self.observation_dim)
        normalized_actions_flat = act_norm.reshape(-1, self.action_dim)
        normalized_target_delta_flat = target_delta_norm.reshape(-1, self.observation_dim)
        normalized_input_flat = torch.cat([normalized_observations_flat, normalized_actions_flat], dim=-1)
    
        predicted_normalized_delta_flat = functional_call(self.model, parameters, (normalized_input_flat,))
        loss = torch.mean((predicted_normalized_delta_flat - normalized_target_delta_flat) ** 2)
        return loss
        
    def predict_next_state(self, observation, action):
        self._assert_normalized()
        normalized_observations = (observation - self.observations_mean) / (self.observations_std + 1e-8)
        normalized_actions = (action - self.actions_mean) / (self.actions_std + 1e-8)
            
        normalized_observations_flat = normalized_observations.reshape(-1, self.observation_dim)
        normalized_actions_flat = normalized_actions.reshape(-1, self.action_dim)
        
        normalized_input_flat = torch.cat([normalized_observations_flat, normalized_actions_flat], dim=-1)
        
        predicted_normalized_delta_flat = self.model(normalized_input_flat)
        
        observation_flat = observation.reshape(-1, self.observation_dim)
        unnormalized_delta_flat = predicted_normalized_delta_flat * (self.delta_std + 1e-8) + self.delta_mean
        
        next_observation_flat = observation_flat + unnormalized_delta_flat
        next_observation = next_observation_flat.reshape(*observation.shape[:-1], self.observation_dim)
        
        return next_observation
            
    def predict_next_state_with_parameters(self, observation, action, parameters=None):
        if parameters is None:
            raise ValueError("parameters must be provided for predict_next_state_with_parameters")
        
        self._assert_normalized()
        normalized_observations = (observation - self.observations_mean) / (self.observations_std + 1e-8)
        normalized_actions = (action - self.actions_mean) / (self.actions_std + 1e-8)
            
        normalized_observations_flat = normalized_observations.reshape(-1, self.observation_dim)
        normalized_actions_flat = normalized_actions.reshape(-1, self.action_dim)
        
        normalized_input_flat = torch.cat([normalized_observations_flat, normalized_actions_flat], dim=-1)
        
        predicted_normalized_delta_flat = functional_call(self.model, parameters, (normalized_input_flat,))
    
        observation_flat = observation.reshape(-1, self.observation_dim)
        unnormalized_delta_flat = predicted_normalized_delta_flat * (self.delta_std + 1e-8) + self.delta_mean
        
        next_observation_flat = observation_flat + unnormalized_delta_flat
        next_observation = next_observation_flat.reshape(*observation.shape[:-1], self.observation_dim)
        
        return next_observation
        
    def get_parameter_dict(self):
        parameter_dict = OrderedDict(self.model.named_parameters())
        return parameter_dict

    # TODO OCEAN This was my inner updater, i will move it out later
    def compute_adapted_params(self, observations, actions, next_observations) -> dict:
        """Run inner-loop gradient descent on support data and return adapted parameters θ′."""
        
        # Ensure we have valid normalization stats before computing losses
        self._assert_normalized()

        # Convert numpy inputs to torch on the correct device
        device = self.observations_mean.device
        if not torch.is_tensor(observations):
            observations = torch.as_tensor(observations, dtype=torch.float32, device=device)
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
        if not torch.is_tensor(next_observations):
            next_observations = torch.as_tensor(next_observations, dtype=torch.float32, device=device)
        
        # Start from the current meta-parameters θ (do NOT mutate self.model in-place)
        parameters = self.get_parameter_dict()
        target_delta = next_observations - observations
        
        for _ in range(self.inner_steps):
            loss = self.compute_normalized_delta_loss_with_parameters(observations, actions, target_delta=target_delta, parameters=parameters)
            
            
            parameter_tensors = tuple(parameters.values())

            grads = torch.autograd.grad(
                loss,
                parameter_tensors,
                create_graph=False,   # rollout/eval does not need higher-order grads
                allow_unused=True,
            )

            updated = OrderedDict()
            for (name, param), grad in zip(parameters.items(), grads):
                clean_grad = torch.zeros_like(param) if grad is None else grad
                updated[name] = param - self.inner_lr * clean_grad

            parameters = updated

        # Return the adapted parameters θ′
        return parameters
        
        
    
