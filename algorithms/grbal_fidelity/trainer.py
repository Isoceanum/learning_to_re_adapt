import os
import torch
from algorithms.base_trainer import BaseTrainer

import time
import numpy as np
import torch

from algorithms.grbal_fidelity.buffer import ReplayBuffer
from algorithms.grbal_fidelity.dynamics_model import DynamicsModel
from algorithms.grbal_fidelity.meta_trainer import MetaTrainer
from algorithms.grbal_fidelity.planner import CrossEntropyMethodPlanner, RandomShootingPlanner

from utils.seed import set_seed

class GrBALFidelityTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        
        self.env = self._make_train_env()
        self.dynamics_model = self._make_dynamics_model().to(self.device)
        self.buffer = self._make_buffer()
        self.planner =self._make_planner()
        self.meta_trainer = self._make_meta_trainer()
        
    def _make_dynamics_model(self):
        hidden_sizes = self.train_config.get("hidden_sizes")
        learning_rate = float(self.train_config.get("learning_rate"))
        
        train_epochs = int(self.train_config.get("train_epochs"))
        valid_split_ratio = float(self.train_config.get("valid_split_ratio"))
        patience = int(self.train_config.get("patience"))
        
        return DynamicsModel(
            observation_dim = self.env.observation_space.shape[0], 
            action_dim = self.env.action_space.shape[0], 
            hidden_sizes = hidden_sizes, 
            learning_rate = learning_rate,
            train_epochs = train_epochs,
            valid_split_ratio = valid_split_ratio,
            patience = patience
            )
   
    def _make_planner(self):
        planner_type = self.train_config.get("planner") 
        horizon = int(self.train_config.get("horizon"))
        n_candidates = int(self.train_config.get("n_candidates"))
        discount = float(self.train_config.get("discount"))
        
        base_env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(base_env, "get_model_reward_fn"):
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")

        reward_fn = base_env.get_model_reward_fn()
        
        action_space = self.env.action_space
        act_low = action_space.low
        act_high = action_space.high
        
        if planner_type == "cem":
            return CrossEntropyMethodPlanner(
            dynamics_fn=self.dynamics_model.predict_next_state_with_parameters,
            reward_fn=reward_fn,
            horizon=horizon,
            n_candidates=n_candidates,
            act_low=act_low,
            act_high=act_high,
            discount=discount,
            seed=self.train_seed,
            device = self.device
        )
            
        if planner_type == "rs":
            return RandomShootingPlanner(
            dynamics_fn=self.dynamics_model.predict_next_state_with_parameters,
            reward_fn=reward_fn,
            horizon=horizon,
            n_candidates=n_candidates,
            act_low=act_low,
            act_high=act_high,
            discount=discount,
            seed=self.train_seed,
            device = self.device
        )
            
        raise AttributeError(f"Planner type {planner_type} not supported")
    
    def _make_buffer(self):
        total_env_steps = int(self.train_config["total_env_steps"])
        buffer_size = total_env_steps
        observation_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        return ReplayBuffer(max_size = buffer_size, observation_dim = observation_dim, action_dim = action_dim)   
    
    def _make_meta_trainer(self):
        past_window_size = int(self.train_config.get("past_window_size"))
        future_window_size = int(self.train_config.get("future_window_size"))
        meta_batch_size = int(self.train_config.get("meta_batch_size"))
        inner_lr = float(self.train_config.get("inner_lr"))
        inner_steps = int(self.train_config.get("inner_steps"))
        meta_outer_lr = float(self.train_config.get("meta_outer_lr"))
        
        return MetaTrainer(self.dynamics_model, self.buffer, past_window_size, future_window_size, meta_batch_size, inner_lr, inner_steps, meta_outer_lr)
        
    def _collect_rollouts(self, remaining_steps, random_actions):
        max_path_length = int(self.train_config["max_path_length"])
        steps_per_iter = int(self.train_config["steps_per_iter"])
        steps_target = min(steps_per_iter, remaining_steps)

        trajectories = []
        steps_used = 0

        if remaining_steps <= 0:
            return trajectories, steps_used

        while steps_used < steps_target:
            if remaining_steps <= 0:
                break
            
            obs, _ = self.env.reset()
            self.buffer.set_episode_start()

            observations = [obs]
            actions = []
            rewards = []
            next_observations = []
            terminated_flags = []
            truncated_flags = []

            rollout_budget = min(max_path_length, steps_target - steps_used)

            for _step in range(rollout_budget):
                if random_actions:
                    action = self.env.action_space.sample()
                else:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    adapted_params = self._compute_adapted_parameters_from_buffer(self.buffer)
                    action = self.planner.plan(obs_tensor, parameters=adapted_params).detach().cpu().numpy()
                    
                    
                next_obs, reward, terminated, truncated, info = self._step_env(action)
                self.buffer.add(obs, action, next_obs)

                actions.append(action)
                rewards.append(reward)
                next_observations.append(next_obs)
                terminated_flags.append(terminated)
                truncated_flags.append(truncated)
                observations.append(next_obs)
                steps_used += 1
                remaining_steps -= 1
                obs = next_obs

                if terminated or truncated or steps_used >= steps_target:
                    break

            if len(actions) == 0:
                break

            trajectory = {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "next_observations": next_observations,
                "terminated": terminated_flags,
                "truncated": truncated_flags,
            }
            trajectories.append(trajectory)

        return trajectories, steps_used
 
    def _process_batch(self, trajectories):
        observations_batch = []
        actions_batch = []
        next_observations_batch = []
        episode_returns = []
        episode_lengths = []
        
        for trajectory in trajectories:
            observations = np.array(trajectory["observations"])
            actions  = np.array(trajectory["actions"])
            rewards = np.array(trajectory["rewards"])
                  
            observations_batch.append(observations[:-1])
            actions_batch.append(actions)
            next_observations_batch.append(observations[1:])
        
            episode_returns.append(rewards.sum())
            episode_lengths.append(len(rewards))
        
    
        observations_all = np.concatenate(observations_batch, axis=0)
        actions_all = np.concatenate(actions_batch, axis=0)
        next_observations_all = np.concatenate(next_observations_batch, axis=0)
        
        delta_all = next_observations_all - observations_all

        obs_mean = observations_all.mean(axis=0)
        obs_std = observations_all.std(axis=0) + 1e-8

        act_mean = actions_all.mean(axis=0)
        act_std = actions_all.std(axis=0) + 1e-8

        delta_mean = delta_all.mean(axis=0)
        delta_std = delta_all.std(axis=0) + 1e-8    
        
        normalization = {
            "observations_mean": obs_mean,
            "observations_std": obs_std,
            "actions_mean": act_mean,
            "actions_std": act_std,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
        }
        
        batch_results = {
            "observations": observations_all,
            "actions": actions_all,
            "next_observations": next_observations_all,
            "normalization": normalization,
        }
        
        return batch_results
    
    def _train_dynamics(self, processed_batch):    
        observations = processed_batch["observations"]
        actions = processed_batch["actions"]
        next_observations = processed_batch["next_observations"]
        normalization = processed_batch["normalization"]
        
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
        normalization = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in normalization.items()
        }
        
        self.dynamics_model.set_normalization_stats(normalization)
        train_metrics = self.dynamics_model.fit(observations, actions, next_observations)
        return train_metrics           

    def _update_normalization_for_iteration(self, trajectories):
        batch_results = self._process_batch(trajectories)
        normalization = batch_results["normalization"]
        normalization = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
            for k, v in normalization.items()
        }
        self.dynamics_model.set_normalization_stats(normalization)
        
    def _compute_adapted_parameters_from_buffer(self, buffer):
        """
        Compute adapted dynamics parameters θ′ from the most recent
        past_window_size transitions in the *current episode* of the given buffer.

        If there are not enough transitions yet, return None so we fall back
        to the prior θ*.
        """
        past_window_size = int(self.train_config.get("past_window_size"))

        # If we don't have enough transitions in this episode yet, skip adaptation
        if buffer.episode_size() < past_window_size:
            return None

        # Retrieve the last `past_window_size` transitions from the current episode
        obs, act, next_obs = buffer.retrieve_recent_transitions_in_episode(past_window_size)
        
        # Convert to torch tensors on the same device as the dynamics model
        device = self.device
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        act = torch.as_tensor(act, dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

        # Build the support batch expected by InnerUpdater
        support_batch = {
            "obs": obs,
            "act": act,
            "next_obs": next_obs,
        }

        # Use the same inner updater as meta-training to get θ′
        theta_prime = self.meta_trainer.inner_updater.compute_adapted_params(support_batch)

        return theta_prime
        
    def _pretrain_dynamics_model(self):
        total_env_steps = int(self.train_config["total_env_steps"])
        pretrain_epochs = int(self.train_config["pretrain_epochs"])
        trajectories, steps_used = self._collect_rollouts(total_env_steps, True)
        
        if steps_used == 0:
            return steps_used
        
        batch_results = self._process_batch(trajectories)
        last_metrics = None
        
        for _ in range(pretrain_epochs):
            last_metrics = self._train_dynamics(batch_results)
            
        if last_metrics is not None:
            print(f"[PRETRAIN] steps_used={steps_used}, train_loss={last_metrics.get('train_loss')}, val_loss={last_metrics.get('val_loss')}")
        
        return steps_used
            
    def _log_iteration(self, iteration, total_steps, meta_metrics):
        if meta_metrics is not None:
            support_loss_val = meta_metrics.get("support_loss_val")
            query_loss_val = meta_metrics.get("query_loss_val")
        
            print(f"[Iteration {iteration}] total_steps={total_steps}, support_loss_val={support_loss_val}, query_loss_val={query_loss_val}")
        else:
            print(f"[Iteration {iteration}] total_steps={total_steps}")
            
    # We override evaluate_checkpoint to handle cases where it is called before norm stats are computed. 
    def evaluate_checkpoint(self):
        dm = self.dynamics_model
        if any(v is None for v in (dm.observations_mean, dm.observations_std, dm.actions_mean, dm.actions_std, dm.delta_mean, dm.delta_std)):
            stats = self.buffer.compute_normalization_stats()
            stats = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in stats.items()}
            dm.set_normalization_stats(stats)
        super().evaluate_checkpoint()


    # we override the base _evaluate to account for grbal algo during eval/deploy
    def _evaluate(self, episodes, seeds):
        all_rewards = []
        forward_progresses = []
        episode_lengths = []
        eval_start_time = time.time()
        
        for seed in seeds:
            set_seed(seed)
            eval_env = self._make_eval_env(seed=seed)
            seed_rewards = []
            seed_forward = []
            seed_lengths = []
            
            for episode in range(episodes):    
                max_path_length = int(self.train_config["max_path_length"])
                episode_buffer = ReplayBuffer(max_size=max_path_length, observation_dim=self.env.observation_space.shape[0], action_dim=self.env.action_space.shape[0])
                episode_buffer.set_episode_start()

                obs, _ = eval_env.reset()
                com_x_start = None

                done = False
                ep_reward = 0.0
                steps = 0
                last_com_x = None

                while not done:
                    
                    prev_obs = obs
                    obs_t = torch.as_tensor(prev_obs, dtype=torch.float32, device=self.device)
                    adapted_params = self._compute_adapted_parameters_from_buffer(episode_buffer)
                    action = self.planner.plan(obs_t, parameters=adapted_params)
                    
                    if isinstance(action, torch.Tensor):
                        action = action.detach().cpu().numpy()
                        
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                        
                    done = terminated or truncated
                    ep_reward += float(reward)
                    steps += 1
                    
                    episode_buffer.add(prev_obs, action, obs)
                    
                    if com_x_start is None:
                        com_x_start = self._get_forward_position(info)
                    last_com_x = self._get_forward_position(info)
                    
                # Compute forward progress
                fp = last_com_x - com_x_start if (com_x_start is not None and last_com_x is not None) else 0.0
                
                seed_rewards.append(ep_reward)
                seed_forward.append(fp)
                seed_lengths.append(steps)
                all_rewards.append(ep_reward)
                forward_progresses.append(fp)
                episode_lengths.append(steps)
            eval_env.close()
        
        return {
            "time_steps": self._global_env_step_counter,
            "reward_mean": np.mean(all_rewards),
            "reward_std": np.std(all_rewards),
            "forward_progress_mean": np.mean(forward_progresses),
            "forward_progress_std": np.std(forward_progresses),
            "episode_length_mean": np.mean(episode_lengths),
            "elapsed": time.time() - eval_start_time,
        }
           
    def train(self):
        print("Starting GrBAL fidelity training")
        start_time = time.time()
        total_env_steps = int(self.train_config["total_env_steps"])
        meta_epochs_per_iteration = int(self.train_config["meta_epochs_per_iteration"])

        total_steps = 0
        iteration = 0
        
        steps_used = self._pretrain_dynamics_model()
         # We used random actions to collect warmup trajectories, this must be ignored during meta RL hence storing the index in buffer
        self.buffer.set_warmup_end_index()
        total_steps += steps_used
        

        while total_steps < total_env_steps:
            remaining = total_env_steps - total_steps

            trajectories, steps_used = self._collect_rollouts(remaining, False)
            if steps_used == 0:
                break
            
            self._update_normalization_for_iteration(trajectories)
            
            last_meta_metrics = None
            for _ in range(meta_epochs_per_iteration):
                last_meta_metrics = self.meta_trainer.run_outer_iteration()

            total_steps += steps_used
            iteration += 1
            
            self._log_iteration(iteration, total_steps, last_meta_metrics)
        
        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"Training finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")
           
    def save(self):
        save_path = os.path.join(self.output_dir, "model.pt")

        # Grab weights and normalization tensors
        payload = {"state_dict": self.dynamics_model.state_dict()}
        norm_stats = {
            "observations_mean": self.dynamics_model.observations_mean,
            "observations_std": self.dynamics_model.observations_std,
            "actions_mean": self.dynamics_model.actions_mean,
            "actions_std": self.dynamics_model.actions_std,
            "delta_mean": self.dynamics_model.delta_mean,
            "delta_std": self.dynamics_model.delta_std,
        }

        if any(v is None for v in norm_stats.values()):
            raise RuntimeError("Normalization stats are missing; train the dynamics model before saving.")

        # Detach to CPU tensors for portability
        payload["normalization"] = {k: v.detach().cpu() for k, v in norm_stats.items()}

        torch.save(payload, save_path)
        print(f"Dynamics model saved to {save_path}")
        
    def load(self, path):
        model_path = path
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")

        ckpt = torch.load(model_path, map_location="cpu")

        # Restore weights
        state_dict = ckpt.get("state_dict", ckpt)
        self.dynamics_model.load_state_dict(state_dict)

        # Restore normalization stats (required for planning)
        normalization = ckpt.get("normalization")
        if normalization is None:
            raise RuntimeError(
                "Checkpoint is missing normalization stats. Re-train with the updated save() so the stats are stored."
            )
        normalization = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device)for k, v in normalization.items()}
        self.dynamics_model.set_normalization_stats(normalization)

        print(f"Loaded dynamics model from {model_path}")
        return self

