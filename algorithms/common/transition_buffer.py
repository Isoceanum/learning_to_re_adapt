import numpy as np
import torch


class TransitionBuffer:
    def __init__(self, valid_split_ratio, seed):
        self.valid_split_ratio = valid_split_ratio
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.train_observations = []
        self.train_actions = []
        self.train_next_observations = []

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

    def get_trajectories(self, split):
        if split == "eval":
            return self.eval_observations, self.eval_actions, self.eval_next_observations
        return self.train_observations, self.train_actions, self.train_next_observations
        
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
        
  
    
