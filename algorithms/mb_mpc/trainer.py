import os

import numpy as np
from algorithms.base_trainer import BaseTrainer

import torch
import math
import time

from algorithms.mb_mpc.dynamics_model import DynamicsModel
from algorithms.mb_mpc.planner import RandomShootingPlanner, CrossEntropyMethodPlanner, FaithfulCrossEntropyMethodPlanner, MPPIPlanner
from algorithms.common.transition_buffer import TransitionBuffer
from algorithms.mb_mpc import sampler
from algorithms.common.planner import make_planner


class MBMPCKStepTrainer(BaseTrainer):
    def __init__(self, config, output_dir):

        super().__init__(config, output_dir)
        
        self.dynamics_model = self._make_dynamics_model().to(self.device)
        self.planner =self._make_planner()
        self.buffer = self._make_buffer()
        
    def _make_buffer(self):
        valid_split_ratio = float(self.train_config["valid_split_ratio"])
        return TransitionBuffer(valid_split_ratio, self.train_seed)
    
    def _make_dynamics_model(self):
        dynamics_model_config = self.train_config.get("dynamics_model")
        if dynamics_model_config is None: 
            raise AttributeError("Missing dynamics_model config in YAML")

        hidden_sizes = dynamics_model_config.get("hidden_sizes")
        learning_rate = float(dynamics_model_config.get("learning_rate"))
        seed = self.train_seed
        
        return DynamicsModel(self.observation_dim, self.action_dim, hidden_sizes, learning_rate, seed)
    
    
    def _make_planner(self):
        planner_config = self.train_config.get("planner")
        base_env = getattr(self.env, "unwrapped", self.env)
        
        if not hasattr(base_env, "get_model_reward_fn"):
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")
        
        reward_fn = base_env.get_model_reward_fn()
        dynamics_fn = self.dynamics_model.predict_next_state

        return make_planner(planner_config, dynamics_fn, reward_fn, self.env.action_space, self.device, self.train_seed)

    def _batch_predict(self, obs_batch, env_indices, iteration_index):
        if iteration_index == 0:
            return np.stack(
                [self.env.action_space.sample() for _ in env_indices],
                axis=0,
            )

        actions = self.planner.plan_batch(obs_batch)
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        return actions


    def _train_dynamics_for_iteration(self, train_epochs, batch_size, steps_per_epoch, eval_batch_size):
        log_print_every_k_epochs = 5
        rolling_p = 0.99
        eval_loss_ema = None
        eval_loss_ema_prev = None
        
        for _epoch in range(train_epochs):
            epoch_start_time = time.time()
            
            epoch_loss_sum = 0.0
            for _ in range(steps_per_epoch):
                batch_obs, batch_act, batch_next_obs = sampler.sample_transitions(self.buffer, batch_size, "train")
                
                batch_obs = batch_obs.to(self.device)
                batch_act = batch_act.to(self.device)
                batch_next_obs = batch_next_obs.to(self.device)
                
                loss_value = self.dynamics_model.train_on_batch(batch_obs, batch_act, batch_next_obs)
                epoch_loss_sum += loss_value
                    
            avg_epoch_loss = epoch_loss_sum / steps_per_epoch                    
            epoch_time_s = time.time() - epoch_start_time
            should_print = (_epoch % log_print_every_k_epochs == 0) or (_epoch == train_epochs - 1)
            
            
            # --- compute eval loss every epoch (needed for early stopping) ---
            eval_loss = float("nan")
            
            if eval_batch_size > 0 and steps_per_epoch > 0:
                eval_loss_sum = 0.0
                with torch.no_grad():
                    for _ in range(steps_per_epoch):
                        eval_obs_batch, eval_act_batch, eval_next_obs_batch = sampler.sample_transitions(self.buffer, eval_batch_size, "eval")
                        eval_obs_batch = eval_obs_batch.to(self.device)
                        eval_act_batch = eval_act_batch.to(self.device)
                        eval_next_obs_batch = eval_next_obs_batch.to(self.device)
                        eval_delta_batch = eval_next_obs_batch - eval_obs_batch
                        eval_loss_sum += self.dynamics_model.loss(eval_obs_batch, eval_act_batch, eval_delta_batch).item()
                eval_loss = eval_loss_sum / steps_per_epoch
                
            just_initialized_ema = False
            if eval_loss_ema is None and eval_batch_size > 0 and steps_per_epoch > 0:
                eval_loss_ema = 1.5 * eval_loss
                eval_loss_ema_prev = 2.0 * eval_loss
                just_initialized_ema = True
                
            if eval_loss_ema is not None:
                eval_loss_ema = rolling_p * eval_loss_ema + (1.0 - rolling_p) * eval_loss
                
            if (not just_initialized_ema) and (eval_loss_ema_prev is not None) and (eval_loss_ema_prev < eval_loss_ema):
                print(f"Early stopping at epoch {_epoch}: eval_ema worsened ({eval_loss_ema_prev:.6f} -> {eval_loss_ema:.6f})")
                print(f"epoch {_epoch}/{train_epochs}: " f"train={avg_epoch_loss:.6f} " f"eval={eval_loss:.6f} " f"time={epoch_time_s:.2f}s" )
                break
            
            eval_loss_ema_prev = eval_loss_ema
            
            if should_print:
                print(f"epoch {_epoch}/{train_epochs}: " f"train={avg_epoch_loss:.6f} " f"eval={eval_loss:.6f} " f"time={epoch_time_s:.2f}s" )
               
        
    def train(self):
        print("Starting MB-MPC training")
        start_time = time.time()
        
        max_path_length = int(self.environment_config["max_episode_length"]) # max steps per episode
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        iterations = int(self.train_config["iterations"])
        train_epochs = int(self.train_config["train_epochs"])
        batch_size = int(self.train_config["batch_size"]) 
        
        for iteration_index in range(iterations):
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            
            self.collect_steps(iteration_index, steps_per_iteration, max_path_length, self.buffer.add_trajectory)
            num_train_transitions = sum(len(ep) for ep in self.buffer.train_observations)

            mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta = self.buffer.get_normalization_stats()
            self.dynamics_model.update_normalization_stats(mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta)
      
            steps_per_epoch = math.ceil(num_train_transitions / batch_size)

            num_eval_transitions = sum(len(ep) for ep in self.buffer.eval_observations)
            self._train_dynamics_for_iteration(train_epochs, batch_size, steps_per_epoch, num_eval_transitions)
            

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
        import torch
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = self.planner.plan(obs_t)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
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
