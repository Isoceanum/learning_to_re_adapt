import numpy as np
import torch


class TransitionBuffer:
    def __init__(self, valid_split_ratio = 0.1, seed = 42):
        self.valid_split_ratio = valid_split_ratio
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        
        self.eval_observations = []
        self.eval_actions = []
        self.eval_next_observations = []
        self.eval_base_predicted_next_observations = []
        
        self.running_mean_obs = None
        self.running_mean_act = None
        self.running_mean_base_pred_delta = None
        self.running_mean_r_delta = None

        self.running_sum_sq_dev_obs = None
        self.running_sum_sq_dev_act = None
        self.running_sum_sq_dev_base_pred_delta = None
        self.running_sum_sq_dev_r_delta = None

        self.normalizer_count = 0

        self.reset()

    def reset(self):
        # Storage placeholders used for training (raw, unnormalized)
        self.train_observations = []
        self.train_actions = []
        self.train_next_observations = []
        self.train_base_predicted_next_observations = []
        return

        # Storage placeholders used for eval and early stop to prevent overfitting (raw, unnormalized)
        self.eval_observations = []
        self.eval_actions = []
        self.eval_next_observations = []
        self.eval_base_predicted_next_observations = []
        
         

        # Running stats are recomputed from scratch per iteration using TRAIN trajectories only

        
    def _assert_episode_lengths(self, observations, actions, next_observations, base_predicted_next_observations):
        if not (len(observations) == len(actions) == len(next_observations) == len(base_predicted_next_observations)):
            raise ValueError(
                "Length mismatch between: observations, actions, next_observations, base_predicted_next_observations"
            )

    def _derive_base_pred_delta_and_r_delta(
        self, observations, actions, next_observations, base_predicted_next_observations
    ):
        # base_pred_delta = f_base(s_t, a_t) - s_t
        base_pred_delta = base_predicted_next_observations - observations
        # r_delta = (s_{t+1} - s_t) - base_pred_delta
        r_delta = (next_observations - observations) - base_pred_delta
        return base_pred_delta, r_delta

    def _update_running_stats(self, batch_array, running_mean, running_sum_sq_dev, current_count):
        """
        Update (mean, sum_sq_dev, count) for a stream using the same parallel
        batch-combine math as the previous implementation (Chan/Welford-style).
        """
        batch_count = int(batch_array.shape[0])
        if batch_count == 0:
            return running_mean, running_sum_sq_dev, current_count

        batch_mean = batch_array.mean(axis=0)
        if current_count == 0:
            running_mean = batch_mean
            running_sum_sq_dev = ((batch_array - batch_mean) ** 2).sum(axis=0)
            return running_mean, running_sum_sq_dev, batch_count

        total_count = current_count + batch_count
        delta_mean = batch_mean - running_mean
        running_mean = running_mean + delta_mean * (batch_count / total_count)
        running_sum_sq_dev = (
            running_sum_sq_dev
            + ((batch_array - batch_mean) ** 2).sum(axis=0)
            + (delta_mean ** 2) * (current_count * batch_count / total_count)
        )
        return running_mean, running_sum_sq_dev, total_count

    def _update_norm_stats_from_trajectory(self, observations, actions, next_observations, base_predicted_next_observations):
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        next_observations = np.asarray(next_observations, dtype=np.float32)
        base_predicted_next_observations = np.asarray(base_predicted_next_observations, dtype=np.float32)

        batch_count = int(observations.shape[0])
        if batch_count == 0:
            return

        # New stats (derived on the fly; deltas are NOT stored in the buffer)
        base_pred_delta, r_delta = self._derive_base_pred_delta_and_r_delta(
            observations, actions, next_observations, base_predicted_next_observations
        )

        current_count = self.normalizer_count

        # Update each stream with identical math/weighting, then set count once.
        self.running_mean_obs, self.running_sum_sq_dev_obs, new_count = self._update_running_stats(
            observations, self.running_mean_obs, self.running_sum_sq_dev_obs, current_count
        )
        self.running_mean_act, self.running_sum_sq_dev_act, _ = self._update_running_stats(
            actions, self.running_mean_act, self.running_sum_sq_dev_act, current_count
        )
        self.running_mean_base_pred_delta, self.running_sum_sq_dev_base_pred_delta, _ = self._update_running_stats(
            base_pred_delta, self.running_mean_base_pred_delta, self.running_sum_sq_dev_base_pred_delta, current_count,
        )
        self.running_mean_r_delta, self.running_sum_sq_dev_r_delta, _ = self._update_running_stats(
            r_delta, self.running_mean_r_delta, self.running_sum_sq_dev_r_delta, current_count
        )
        self.normalizer_count = new_count

    def _choose_split(self):
        if len(self.train_observations) == 0: return "train"
        if len(self.eval_observations) == 0: return "eval"
    
        return "eval" if self.rng.random() < self.valid_split_ratio else "train"
        
    def add_trajectory(self, observations, actions, next_observations, base_predicted_next_observations):
        self._assert_episode_lengths(observations, actions, next_observations, base_predicted_next_observations)
        
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        next_observations = np.asarray(next_observations, dtype=np.float32)
        base_predicted_next_observations = np.asarray(base_predicted_next_observations, dtype=np.float32)
        
        split = self._choose_split()
        
        if split == "eval":
            self.eval_observations.append(observations)
            self.eval_actions.append(actions)
            self.eval_next_observations.append(next_observations)
            self.eval_base_predicted_next_observations.append(base_predicted_next_observations)
        else:
            self.train_observations.append(observations)
            self.train_actions.append(actions)
            self.train_next_observations.append(next_observations)
            self.train_base_predicted_next_observations.append(base_predicted_next_observations)
            self._update_norm_stats_from_trajectory(
                observations, actions, next_observations, base_predicted_next_observations
            )
        
    def sample_transitions(self, batch_size, split):
        if split == "eval":
            observations = self.eval_observations
            actions = self.eval_actions
            next_observations = self.eval_next_observations
            base_predicted_next_observations = self.eval_base_predicted_next_observations
        else:
            observations = self.train_observations
            actions = self.train_actions
            next_observations = self.train_next_observations
            base_predicted_next_observations = self.train_base_predicted_next_observations
            
        if len(observations) == 0: raise RuntimeError(f"No episodes available for split='{split}'")
        
        observations_batch = []
        actions_batch = []
        next_observations_batch = []
        base_predicted_next_observations_batch = []
        
        for _ in range(batch_size): # Loop to collect batch_size transitions
            episode_index = self.rng.integers(0, len(observations)) # Sample random episode from buffer
            episode_length = len(observations[episode_index]) 
            step_index = self.rng.integers(0, episode_length) # Sample random step from episode
            
            observations_batch.append(observations[episode_index][step_index])
            actions_batch.append(actions[episode_index][step_index])
            next_observations_batch.append(next_observations[episode_index][step_index])
            base_predicted_next_observations_batch.append(base_predicted_next_observations[episode_index][step_index])
        
        # Convert into torchs
        observations_batch = torch.as_tensor(np.asarray(observations_batch), dtype=torch.float32)
        actions_batch = torch.as_tensor(np.asarray(actions_batch), dtype=torch.float32)
        next_observations_batch = torch.as_tensor(np.asarray(next_observations_batch), dtype=torch.float32)
        base_predicted_next_observations_batch = torch.as_tensor(
            np.asarray(base_predicted_next_observations_batch), dtype=torch.float32
        )

        return observations_batch, actions_batch, next_observations_batch, base_predicted_next_observations_batch
        
    def get_normalization_stats(self):
        epsilon = 1e-8

        if self.normalizer_count <= 0:
            raise RuntimeError("Normalization stats are not initialized yet.")

        mean_obs = self.running_mean_obs
        mean_act = self.running_mean_act
        mean_base_pred_delta = self.running_mean_base_pred_delta
        mean_r_delta = self.running_mean_r_delta

        denom = float(self.normalizer_count)
        std_obs = np.sqrt(self.running_sum_sq_dev_obs / denom) + epsilon
        std_act = np.sqrt(self.running_sum_sq_dev_act / denom) + epsilon
        std_base_pred_delta = np.sqrt(self.running_sum_sq_dev_base_pred_delta / denom) + epsilon
        std_r_delta = np.sqrt(self.running_sum_sq_dev_r_delta / denom) + epsilon

        return (
            mean_obs,
            std_obs,
            mean_act,
            std_act,
            mean_base_pred_delta,
            std_base_pred_delta,
            mean_r_delta,
            std_r_delta,
        )
        
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
    
    
    
    
