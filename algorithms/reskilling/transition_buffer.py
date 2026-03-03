import numpy as np
import torch


class TransitionBuffer:
    def __init__(self, valid_split_ratio = 0.1, seed = 42):
        self.valid_split_ratio = valid_split_ratio
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Storage placeholders used for training
        self.train_observations = []
        self.train_actions = []
        self.train_next_observations = []

        # Storage placeholders used for eval and early stop to prevent overfitting
        self.eval_observations = []
        self.eval_actions = []
        self.eval_next_observations = []
        
        self.running_mean_obs = None
        self.running_mean_delta = None
        self.running_mean_act = None
        
        self.running_sum_sq_dev_obs = None
        self.running_sum_sq_dev_act = None
        self.running_sum_sq_dev_delta = None
        
        self.normalizer_count = 0
        
    def _assert_episode_lengths(self, observations, actions, next_observations):
        if not (len(observations) == len(actions) == len(next_observations)): 
            raise ValueError(f"Length mismatch between: observations, actions and next_observations")

    def _update_norm_stats_from_trajectory(self, observations, actions, next_observations):
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        next_observations = np.asarray(next_observations, dtype=np.float32)
        deltas = next_observations - observations
        batch_count = observations.shape[0]
        
        if batch_count == 0:
            return
        
        batch_mean_obs = observations.mean(axis=0)
        batch_mean_act = actions.mean(axis=0)
        batch_mean_delta = deltas.mean(axis=0)
        
        if self.normalizer_count == 0:
            self.running_mean_obs = batch_mean_obs
            self.running_mean_act = batch_mean_act
            self.running_mean_delta = batch_mean_delta
            
            self.running_sum_sq_dev_obs = ((observations - batch_mean_obs) ** 2).sum(axis=0)
            self.running_sum_sq_dev_act = ((actions - batch_mean_act) ** 2).sum(axis=0)
            self.running_sum_sq_dev_delta = ((deltas - batch_mean_delta) ** 2).sum(axis=0)
            
            self.normalizer_count = batch_count
            return
    
        total_count = self.normalizer_count + batch_count
        delta_mean_obs = batch_mean_obs - self.running_mean_obs
        delta_mean_act = batch_mean_act - self.running_mean_act
        delta_mean_delta = batch_mean_delta - self.running_mean_delta
        
        self.running_mean_obs = self.running_mean_obs + delta_mean_obs * (batch_count / total_count)
        self.running_mean_act = self.running_mean_act + delta_mean_act * (batch_count / total_count)
        self.running_mean_delta = self.running_mean_delta + delta_mean_delta * (batch_count / total_count)
        
        self.running_sum_sq_dev_obs = self.running_sum_sq_dev_obs + ((observations - batch_mean_obs) ** 2).sum(axis=0) + (delta_mean_obs ** 2) * (self.normalizer_count * batch_count / total_count)
        self.running_sum_sq_dev_act = self.running_sum_sq_dev_act + ((actions - batch_mean_act) ** 2).sum(axis=0) + (delta_mean_act ** 2) * (self.normalizer_count * batch_count / total_count)
        self.running_sum_sq_dev_delta = self.running_sum_sq_dev_delta + ((deltas - batch_mean_delta) ** 2).sum(axis=0) + (delta_mean_delta ** 2) * (self.normalizer_count * batch_count / total_count)
        self.normalizer_count = total_count

    def _choose_split(self):
        if len(self.train_observations) == 0: return "train"
        if len(self.eval_observations) == 0: return "eval"
    
        return "eval" if self.rng.random() < self.valid_split_ratio else "train"
        
    def add_trajectory(self, observations, actions, next_observations):
        self._assert_episode_lengths(observations, actions, next_observations)
        
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        next_observations = np.asarray(next_observations, dtype=np.float32)
        
        split = self._choose_split()
        
        if split == "eval":
            self.eval_observations.append(observations)
            self.eval_actions.append(actions)
            self.eval_next_observations.append(next_observations)
        else:
            self.train_observations.append(observations)
            self.train_actions.append(actions)
            self.train_next_observations.append(next_observations)
            self._update_norm_stats_from_trajectory(observations, actions, next_observations)
        
    def sample_transitions(self, batch_size, split):
        if split == "eval":
            observations = self.eval_observations
            actions = self.eval_actions
            next_observations = self.eval_next_observations
        else:
            observations = self.train_observations
            actions = self.train_actions
            next_observations = self.train_next_observations
            
        if len(observations) == 0: raise RuntimeError(f"No episodes available for split='{split}'")
        
        observations_batch = []
        actions_batch = []
        next_observations_batch = []
        
        for _ in range(batch_size): # Loop to collect batch_size transitions
            episode_index = self.rng.integers(0, len(observations)) # Sample random episode from buffer
            episode_length = len(observations[episode_index]) 
            step_index = self.rng.integers(0, episode_length) # Sample random step from episode
            
            observations_batch.append(observations[episode_index][step_index])
            actions_batch.append(actions[episode_index][step_index])
            next_observations_batch.append(next_observations[episode_index][step_index])
        
        # Convert into torchs
        observations_batch = torch.as_tensor(np.asarray(observations_batch), dtype=torch.float32)
        actions_batch = torch.as_tensor(np.asarray(actions_batch), dtype=torch.float32)
        next_observations_batch = torch.as_tensor(np.asarray(next_observations_batch), dtype=torch.float32)

        return observations_batch, actions_batch, next_observations_batch
    
    
    def get_normalization_stats(self):
        epsilon = 1e-8

        if self.normalizer_count <= 0:
            raise RuntimeError("Normalization stats are not initialized yet.")

        mean_obs = self.running_mean_obs
        mean_act = self.running_mean_act
        mean_delta = self.running_mean_delta

        denom = float(self.normalizer_count)
        std_obs = np.sqrt(self.running_sum_sq_dev_obs / denom) + epsilon
        std_act = np.sqrt(self.running_sum_sq_dev_act / denom) + epsilon
        std_delta = np.sqrt(self.running_sum_sq_dev_delta / denom) + epsilon

        return mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta
        
        
    def sample_k_step_batch(self, k_step, batch_size, split):
        if split == "eval":
            observations = self.eval_observations
            actions = self.eval_actions
            next_observations = self.eval_next_observations
        else:
            observations = self.train_observations
            actions = self.train_actions
            next_observations = self.train_next_observations

        if len(observations) == 0:
            raise RuntimeError(f"No episodes available for split='{split}'")

        horizon = k_step
        valid_indices = [idx for idx, ep in enumerate(observations) if len(ep) >= horizon]
        if not valid_indices:
            raise RuntimeError(f"No episodes long enough for horizon {horizon} in split='{split}'")

        obs_batch = []
        action_batch = []
        target_batch = []

        for _ in range(batch_size):
            episode_index = valid_indices[self.rng.integers(0, len(valid_indices))]
            episode_obs = observations[episode_index]
            episode_act = actions[episode_index]
            episode_next_obs = next_observations[episode_index]

            max_start = len(episode_obs) - horizon
            start_index = self.rng.integers(0, max_start + 1) if max_start > 0 else 0

            obs_batch.append(episode_obs[start_index])
            action_batch.append(episode_act[start_index:start_index + horizon])
            target_batch.append(episode_next_obs[start_index:start_index + horizon])

        obs_batch = torch.as_tensor(np.asarray(obs_batch), dtype=torch.float32)
        action_batch = torch.as_tensor(np.asarray(action_batch), dtype=torch.float32)
        target_batch = torch.as_tensor(np.asarray(target_batch), dtype=torch.float32)

        return obs_batch, action_batch, target_batch
    
        
    
    
    
