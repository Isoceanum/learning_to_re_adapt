import os
import torch
from algorithms.base_trainer import BaseTrainer

import time
import numpy as np
import torch

from algorithms.mb_mpc_fidelity.dynamics_model import DynamicsModel
from algorithms.mb_mpc_fidelity.planner import CrossEntropyMethodPlanner, RandomShootingPlanner

class MBMPCFidelityTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        
        self.global_trajectories = []
        self.env = self._make_train_env()
        self.dynamics_model = self._make_dynamics_model().to(self.device)
        self.planner =self._make_planner()
        
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
            dynamics_fn=self.dynamics_model.predict_next_state,
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
            dynamics_fn=self.dynamics_model.predict_next_state,
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
    
    def _collect_rollouts(self, remaining_steps, random_actions):
        max_path_length = int(self.train_config["max_path_length"])
        steps_per_iter = int(self.train_config["steps_per_iter"])
        steps_target = min(steps_per_iter, remaining_steps)

        trajectories = []
        steps_used = 0

        if steps_target <= 0:
            return trajectories, steps_used

        while steps_used < steps_target:
            obs, _ = self.env.reset()

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
                    action = self.planner.plan(obs_tensor).detach().cpu().numpy()
                    
                next_obs, reward, terminated, truncated, info = self._step_env(action)

                actions.append(action)
                rewards.append(reward)
                next_observations.append(next_obs)
                terminated_flags.append(terminated)
                truncated_flags.append(truncated)
                observations.append(next_obs)
                steps_used += 1
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
    
    def _log_iteration(self, iteration, batch_results, total_steps, train_metrics, elapsed):
        train_loss = train_metrics.get('train_loss')
        val_loss = train_metrics.get('val_loss')
        print(f"[Iteration {iteration}] total_steps={total_steps}, train_loss={train_loss}, val_loss={val_loss}, elapsed={elapsed:.2f}")
             
    def _pretrain_dynamics_model(self):
        total_env_steps = int(self.train_config["total_env_steps"])
        pretrain_epochs = int(self.train_config["pretrain_epochs"])
        trajectories, steps_used = self._collect_rollouts(total_env_steps, True)
        self.global_trajectories.extend(trajectories)
        
        if steps_used == 0:
            return steps_used
        
        batch_results = self._process_batch(self.global_trajectories)
        last_metrics = None
        
        for _ in range(pretrain_epochs):
            last_metrics = self._train_dynamics(batch_results)
            
        if last_metrics is not None:
            print(f"[PRETRAIN] steps_used={steps_used}, train_loss={last_metrics.get('train_loss')}, val_loss={last_metrics.get('val_loss')}")
        
        return steps_used
        
    def train(self):
        print("Starting MB-MPC fidelity training")
        start_time = time.time()
        total_env_steps = int(self.train_config["total_env_steps"])
        
        total_steps = 0
        iteration = 0
        
        steps_used = self._pretrain_dynamics_model()
        total_steps += steps_used

        while total_steps < total_env_steps:
            iteration_start_time = time.time() 
            remaining = total_env_steps - total_steps            
            trajectories, steps_used = self._collect_rollouts(remaining, False)
            
            if steps_used == 0:
                break
            
            self.global_trajectories.extend(trajectories)
            batch_results = self._process_batch(self.global_trajectories)
            total_steps += steps_used
            train_metrics = self._train_dynamics(batch_results)
            
            iteration_elapsed = time.time() - iteration_start_time 

            self._log_iteration(iteration, batch_results, total_steps, train_metrics, iteration_elapsed)
            iteration += 1
        
        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"Training finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")
        
    # We override evaluate_checkpoint to handle cases where it is called before norm stats are computed. 
    def evaluate_checkpoint(self):
        dm = self.dynamics_model
        if any(v is None for v in (dm.observations_mean, dm.observations_std, dm.actions_mean, dm.actions_std, dm.delta_mean, dm.delta_std)):
            stats = self.buffer.compute_normalization_stats()
            stats = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in stats.items()}
            dm.set_normalization_stats(stats)
        super().evaluate_checkpoint()

    def predict(self, obs):
        import torch
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = self.planner.plan(obs_t)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return action

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
    
    