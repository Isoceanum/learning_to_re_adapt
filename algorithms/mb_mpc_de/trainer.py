from collections import deque
import os

import numpy as np
from algorithms.base_trainer import BaseTrainer

import torch
import math
import time

from algorithms.mb_mpc_de.dynamics_model import DynamicsModel
from algorithms.mb_mpc_de.planner import RandomShootingPlanner
from algorithms.mb_mpc_de.transition_buffer import TransitionBuffer


class MBMPCDETrainer(BaseTrainer):
    def __init__(self, config, output_dir):

        super().__init__(config, output_dir)
        
        self.env = self._make_train_env()
        self.dynamics_model = self._make_dynamics_model().to(self.device)
        self.planner =self._make_planner()
        self.buffer = self._make_buffer()
        
        self.support_window_size = int(self.train_config["support_window_size"])
        self.inner_learning_rate = float(self.train_config["inner_learning_rate"])
        self.use_online_adaptation = self.train_config["use_online_adaptation"]
        
        self.eval_support_window = None
        self.eval_last_obs = None
        self.eval_last_action = None
        
        
    def _make_buffer(self):
        valid_split_ratio = float(self.train_config["valid_split_ratio"])
        return TransitionBuffer(valid_split_ratio, self.train_seed)
    
    def _make_dynamics_model(self):
        dynamics_model_config = self.train_config.get("dynamics_model")
        if dynamics_model_config is None: raise AttributeError("Missing dynamics_model config in YAML")
    
        observation_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        hidden_sizes = dynamics_model_config.get("hidden_sizes")
        learning_rate = float(dynamics_model_config.get("learning_rate"))
        seed = self.train_seed
        
        return DynamicsModel(observation_dim, action_dim, hidden_sizes, learning_rate, seed)

    def _make_planner(self):
        planner_config = self.train_config.get("planner")
        if planner_config is None:
            raise AttributeError("Missing planner config in YAML")
        
        planner_type = planner_config.get("type")         
        horizon = int(planner_config.get("horizon"))
        n_candidates = int(planner_config.get("n_candidates"))
        discount = float(planner_config.get("discount"))

        base_env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(base_env, "get_model_reward_fn"):
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")

        reward_fn = base_env.get_model_reward_fn()
        
        action_space = self.env.action_space
        act_low = action_space.low
        act_high = action_space.high
        
        if planner_type == "rs":
            return RandomShootingPlanner(self.dynamics_model.predict_next_state_with_parameters, reward_fn, horizon, n_candidates, act_low, act_high, self.device, discount)
            
        raise AttributeError(f"Planner type {planner_type} not supported")
    
    def _collect_steps(self, iteration_index, steps_target, max_path_length):
        steps_collected_this_iteration = 0
            
        log_collect_start_time = time.time()
        log_episodes = 0
        log_episode_forward_progress = []
        log_episode_velocity = []
        log_episode_returns = []
        
        while steps_collected_this_iteration < steps_target:
            obs, _ = self.env.reset()
            log_episodes += 1
            
            episode_return = 0.0
            episode_x_start = None
            episode_x_last = None
            episode_velocity = 0.0

            episode_steps = 0          
            episode_obs = []
            episode_act = []
            episode_next_obs = []
            
            while episode_steps < max_path_length:
                if iteration_index == 0:
                    action = self.env.action_space.sample()
                else:
                    parameter_dict = self.dynamics_model.get_parameter_dict()
                    if len(episode_act) >= self.support_window_size and self.use_online_adaptation:
                        
                        
                        
                        support_obs_np = np.stack(episode_obs[-self.support_window_size:], axis=0)
                        support_act_np = np.stack(episode_act[-self.support_window_size:], axis=0)
                        support_next_obs_np = np.stack(episode_next_obs[-self.support_window_size:], axis=0)
                        
                        support_obs = torch.as_tensor(support_obs_np, dtype=torch.float32, device=self.device)
                        support_act = torch.as_tensor(support_act_np, dtype=torch.float32, device=self.device)
                        support_next_obs = torch.as_tensor(support_next_obs_np, dtype=torch.float32, device=self.device)
                        parameter_dict = self.dynamics_model.compute_adapted_params(support_obs, support_act, support_next_obs, self.inner_learning_rate)
                        
                    action = self.planner.plan(obs, parameters=parameter_dict)
                    
                    if torch.is_tensor(action):
                        action = action.detach().cpu().numpy()
                    
                next_obs, reward, terminated, truncated, info = self._step_env(action)
                episode_return += float(reward)
                
                x_position = float(self._get_forward_position(info))
                if episode_x_start is None:
                    episode_x_start = x_position
                episode_x_last = x_position
                
                episode_velocity += self._get_x_velocity(info)
                
                episode_obs.append(obs)
                episode_act.append(action)
                episode_next_obs.append(next_obs)
                
                obs = next_obs
            
                episode_steps += 1
                steps_collected_this_iteration += 1
                            
                if steps_collected_this_iteration >= steps_target:
                    break
                
                if terminated or truncated:
                    break
                
            self.buffer.add_trajectory(episode_obs, episode_act, episode_next_obs)
            log_episode_forward_progress.append(float(episode_x_last - episode_x_start))
            log_episode_velocity.append(float(episode_velocity))
            log_episode_returns.append(float(episode_return))
                    
        collect_stats = {
            "log_episodes": log_episodes,
            "log_collect_time":  time.time() - log_collect_start_time, 
            "steps_collected_this_iteration": steps_collected_this_iteration,
            "avg_reward": sum(log_episode_returns) / max(1, len(log_episode_returns)),
            "avg_forward_progress": sum(log_episode_forward_progress) / max(1, len(log_episode_forward_progress)),
            "avg_velocity": sum(log_episode_velocity) / max(1, len(log_episode_velocity)),
        }
        
        return collect_stats
        
    def _train_dynamics_for_iteration(self, train_epochs, batch_size, steps_per_epoch, eval_batch):
        eval_obs, eval_act, eval_delta = eval_batch
        log_print_every_k_epochs = 5
        
        for _epoch in range(train_epochs):
            epoch_start_time = time.time()
            
            epoch_loss_sum = 0.0
            for _ in range(steps_per_epoch):
                batch_obs, batch_act, batch_next_obs = self.buffer.sample_transitions(batch_size, "train")
                
                batch_obs = batch_obs.to(self.device)
                batch_act = batch_act.to(self.device)
                batch_next_obs = batch_next_obs.to(self.device)
                
                loss_value = self.dynamics_model.train_on_batch(batch_obs, batch_act, batch_next_obs)
                epoch_loss_sum += loss_value
                    
            avg_epoch_loss = epoch_loss_sum / steps_per_epoch                    
            epoch_time_s = time.time() - epoch_start_time
            should_print = (_epoch % log_print_every_k_epochs == 0) or (_epoch == train_epochs - 1)
            if should_print:
                with torch.no_grad():
                    eval_loss = self.dynamics_model.compute_loss(eval_obs, eval_act, eval_delta).item()
                    
                print(f"epoch {_epoch}/{train_epochs}: " f"train={avg_epoch_loss:.6f} " f"eval={eval_loss:.6f} " f"time={epoch_time_s:.2f}s" )
    
        
    def train(self):
        print(f"Starting MB-MPC-DE training{' (with online adaptation step)' if self.use_online_adaptation else ''}")
        start_time = time.time()
        
        max_path_length = int(self.train_config["max_path_length"])
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        iterations = int(self.train_config["iterations"])
        train_epochs = int(self.train_config["train_epochs"])
        batch_size = int(self.train_config["batch_size"]) 
        
        for iteration_index in range(iterations):
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            
            collect_stats = self._collect_steps(iteration_index, steps_per_iteration, max_path_length)
            num_train_transitions = sum(len(ep) for ep in self.buffer.train_observations)
            avg_reward = collect_stats["avg_reward"]
            avg_forward_progress = collect_stats["avg_forward_progress"]
            avg_velocity = collect_stats["avg_velocity"]
            steps_collected_this_iteration = collect_stats["steps_collected_this_iteration"]
            log_collect_time = collect_stats["log_collect_time"]
            log_episodes = collect_stats["log_episodes"]
            
            print(f"collect: dataset={num_train_transitions} " f"steps={steps_collected_this_iteration} " f"episodes={log_episodes} " f"avg_rew={avg_reward:.3f} " f"avg_fp={avg_forward_progress:.3f} " f"avg_v={avg_velocity:.3f} " f"time={log_collect_time:.1f}s")
            
            mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta = self.buffer.get_normalization_stats()
            self.dynamics_model.update_normalization_stats(mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta)
      
            steps_per_epoch = math.ceil(num_train_transitions / batch_size)
                        
            with torch.no_grad():
                num_eval_transitions = sum(len(ep) for ep in self.buffer.eval_observations)
                eval_batch_size = min(num_eval_transitions, 8192)
                eval_obs, eval_act, eval_next_obs = self.buffer.sample_transitions(eval_batch_size, "eval")
                eval_obs = eval_obs.to(self.device)
                eval_act = eval_act.to(self.device)
                eval_next_obs = eval_next_obs.to(self.device)
                eval_delta = eval_next_obs - eval_obs
                eval_batch = (eval_obs, eval_act, eval_delta)
                
            self._train_dynamics_for_iteration(train_epochs, batch_size, steps_per_epoch, eval_batch)
            

        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"\nTraining finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")
                                         
    def save(self):
        save_path = os.path.join(self.output_dir, "model.pt")
        
        self.dynamics_model._assert_normalization_stats()
        
        norm_stats = {
            "mean_obs": self.dynamics_model.mean_obs.detach().cpu(),
            "std_obs": self.dynamics_model.std_obs.detach().cpu(),
            "mean_act": self.dynamics_model.mean_act.detach().cpu(),
            "std_act": self.dynamics_model.std_act.detach().cpu(),
            "mean_delta": self.dynamics_model.mean_delta.detach().cpu(),
            "std_delta": self.dynamics_model.std_delta.detach().cpu(),
        }
   
        payload = {
            "state_dict": self.dynamics_model.state_dict(),
            "norm_stats": norm_stats,
        }
        torch.save(payload, save_path)
        print(f"Dynamics model saved to {save_path}")
        
    def predict(self, obs):
        if self.eval_support_window is None:
            self.eval_support_window = deque(maxlen=self.support_window_size)
            self.eval_last_obs = None
            self.eval_last_action = None
            
        if self.eval_last_obs is not None and self.eval_last_action is not None:
            self.eval_support_window.append((self.eval_last_obs, self.eval_last_action, obs))
            
            
        params_for_planning = self.dynamics_model.get_parameter_dict()

        if len(self.eval_support_window) >= self.support_window_size and self.use_online_adaptation:
            window_obs, window_act, window_next_obs = zip(*self.eval_support_window)

            support_obs_np = np.stack(window_obs, axis=0)
            support_act_np = np.stack(window_act, axis=0)
            support_next_obs_np = np.stack(window_next_obs, axis=0)

            support_obs = torch.as_tensor(support_obs_np, dtype=torch.float32, device=self.device)
            support_act = torch.as_tensor(support_act_np, dtype=torch.float32, device=self.device)
            support_next_obs = torch.as_tensor(support_next_obs_np, dtype=torch.float32, device=self.device)

            params_for_planning = self.dynamics_model.compute_adapted_params(support_obs, support_act, support_next_obs, self.inner_learning_rate)

        action = self.planner.plan(obs, parameters=params_for_planning)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        self.eval_last_obs = obs
        self.eval_last_action = action
        return action
        

    def load(self, path):
        model_path = path
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")

        # Restore model weights
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.dynamics_model.load_state_dict(state_dict)

        # Restore normalization stats required for planning
        normalization = checkpoint.get("norm_stats")
        if normalization is None:
            raise RuntimeError(
                "Checkpoint is missing normalization stats. Re-train with updated save() so stats are stored."
            )

        # Convert to tensors on correct device (update_normalization_stats handles numpy OR torch, but this is explicit)
        normalization = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in normalization.items()
        }

        self.dynamics_model.update_normalization_stats(
            normalization["mean_obs"], normalization["std_obs"],
            normalization["mean_act"], normalization["std_act"],
            normalization["mean_delta"], normalization["std_delta"],
        )

        print(f"Loaded dynamics model from {model_path}")
        return self
    
    def evaluate_checkpoint(self):
        dm = self.dynamics_model
        if any(v is None for v in (dm.mean_obs, dm.std_obs, dm.mean_act, dm.std_act, dm.mean_delta, dm.std_delta)):
            return
        super().evaluate_checkpoint()

    def _reset_eval_adaptation(self):
        """Optional hook for trainers that keep eval-time adaptation state."""
        self.eval_support_window = None
        self.eval_last_obs = None
        self.eval_last_action = None
