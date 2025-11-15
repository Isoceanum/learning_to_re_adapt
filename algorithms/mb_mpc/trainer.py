import os
import torch
from algorithms.base_trainer import BaseTrainer

from algorithms.mb_mpc.buffer import ReplayBuffer
from algorithms.mb_mpc.dynamics_model import DynamicsModel
from algorithms.mb_mpc.planner import RandomShootingPlanner
from algorithms.mb_mpc.planner import CrossEntropyMethodPlanner

import time
import numpy as np
import torch


class MBMPCTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        
        self.env = self._make_train_env()
        self.buffer = self._make_buffer()
        self.dynamics_model = self._make_dynamics_model()
        self.planner =self._make_planner()
        
    def _make_dynamics_model(self):
        hidden_sizes = self.train_config.get("hidden_sizes")
        learning_rate = float(self.train_config.get("learning_rate"))
        
        return DynamicsModel(
            observation_dim = self.env.observation_space.shape[0], 
            action_dim = self.env.action_space.shape[0], 
            hidden_sizes = hidden_sizes, 
            learning_rate = learning_rate)
        
    def _make_buffer(self):        
        total_env_steps = int(self.train_config["total_env_steps"])
        buffer_size = total_env_steps
        
        return ReplayBuffer(max_size = buffer_size, observation_dim = self.env.observation_space.shape[0], action_dim = self.env.action_space.shape[0])
    
    def _make_planner(self):
        device = self.device
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
            device = device
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
            device = device
        )
            
        raise AttributeError(f"Planner type {planner_type} not supported")
        
    def train(self):
        print("Starting MB-MPC training")
        start_time = time.time()

        # === 1. Random data collection ===
        self.collect_warmup_data()

        # === 2. Initial model pretraining ===
        self.pretrain_dynamics_model()
        
        # === 3. MPC Rollouts ===
        self.run_mpc_rollout()
        
        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"Training finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")
        
        
    def collect_warmup_data(self):
        local_start = time.time()
        total_env_steps = int(self.train_config["total_env_steps"])
        warmup_steps_budget = int(self.train_config["warmup_steps_budget"])
        max_path_length = int(self.train_config["max_path_length"])
        print(f"[COLLECT_WARMUP_DATA] warmup_steps_budget={warmup_steps_budget} | seed={self.train_seed}")
        
        obs, _ = self.env.reset(seed=self.train_seed)
        steps_since_reset = 0
        reset_counter = 0
        for step in range(warmup_steps_budget):
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.buffer.add(obs, action, next_obs)
            obs = next_obs
            steps_since_reset += 1
            if terminated or truncated or steps_since_reset >= max_path_length:
                obs, _ = self.env.reset()
                reset_counter += 1
                steps_since_reset = 0
        
        # === Compute and load normalization stats ===
        norm_stats = self.buffer.compute_normalization_stats()
        self.dynamics_model.set_normalization_stats(norm_stats)
        
        elapsed = time.time() - local_start
        buffer_size = int(self.buffer.current_size)
        print(f"[COLLECT_WARMUP_DATA] episodes={reset_counter} | buffer_size={buffer_size} | elapsed={elapsed:.2f}s \n")
    
    def pretrain_dynamics_model(self):
        local_start = time.time()
        epochs = int(self.train_config["epochs"])
        batch_size = int(self.train_config["batch_size"])
        
        print(f"[PRETRAIN_DYNAMICS] epochs={epochs} | batch_size={batch_size}")

        for epoch in range(epochs):
            epoch_seed = self.train_seed + epoch
            generator = torch.Generator(device="cpu").manual_seed(epoch_seed)
            shuffled_indices = torch.randperm(self.buffer.current_size,  generator=generator)            
            sum_loss = 0.0
            batch_count = 0
            
            for start in range(0, self.buffer.current_size, batch_size):
                idx = shuffled_indices[start:start + batch_size]
                obs_batch, act_batch, next_obs_batch = self.buffer.retrieve_batch(idx)
                loss_val = self.dynamics_model.update(obs_batch, act_batch, next_obs_batch)
                sum_loss += loss_val
                batch_count += 1
                
            avg_loss = sum_loss / max(batch_count, 1)
            if (epoch + 1) % 10 == 0:
                print(f"epoch {epoch+1}/{epochs} | train_loss_avg={avg_loss:.5f} | epoch_seed={epoch_seed}")
                
        elapsed = time.time() - local_start
        print(f"[PRETRAIN_DYNAMICS] finished | elapsed={elapsed:.2f}s \n")
                
    def run_mpc_rollout(self):
        local_start = time.time()
        # === Metric trackers ===
        reward_history = []
        loss_history = []
        
        steps_per_update = int(self.train_config["steps_per_update"])
        max_path_length = int(self.train_config["max_path_length"])
        epochs = int(self.train_config["epochs"])
        batch_size = int(self.train_config["batch_size"])
        total_env_steps = int(self.train_config["total_env_steps"])
        warmup_steps_budget = int(self.train_config["warmup_steps_budget"])
        rollout_steps_budget = max(0, total_env_steps - warmup_steps_budget)
        
        print(f"[MPC_ROLLOUT] rollout_steps_budget={rollout_steps_budget} | max_path_length={max_path_length} | steps_per_update={steps_per_update}")
        
        obs, _ = self.env.reset()
        rollout_steps_used = 0
        steps_since_reset = 0
        total_reward = 0.0
        
        episode_returns = []; 
        episode_lengths = []
        episode_forward_progress = []
        current_ep_return = 0.0
        current_ep_forward = 0.0
        last_forward_position = None
        
        chunk_index = 0
        total_chunks = (rollout_steps_budget + steps_per_update - 1) // steps_per_update if steps_per_update > 0 else 0
        while rollout_steps_used < rollout_steps_budget:
            chunk_start = time.time()
   
            chunk_index += 1
            episodes_before = len(episode_returns)
            steps_this_chunk = min(steps_per_update, rollout_steps_budget - rollout_steps_used)
            chunk_loss_sum = 0.0
            chunk_loss_count = 0
        
            for _ in range(steps_this_chunk):
                action = self.planner.plan(torch.tensor(obs, dtype=torch.float32))
                
                if isinstance(action, torch.Tensor):
                    action = action.detach().cpu().numpy()
                
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                forward_pos = info.get("com_x_position")
                if forward_pos is None:
                    forward_pos = self._get_forward_position(info)
                else:
                    forward_pos = float(forward_pos)
                
                if last_forward_position is not None:
                    delta_forward = forward_pos - last_forward_position
                    current_ep_forward += delta_forward
                last_forward_position = forward_pos
                self.buffer.add(obs, action, next_obs)
                rollout_steps_used += 1
                obs = next_obs
                total_reward += reward
                current_ep_return += reward
                steps_since_reset += 1
                
                if terminated or truncated or steps_since_reset >= max_path_length:
                    episode_returns.append(current_ep_return)
                    episode_lengths.append(steps_since_reset)
                    episode_forward_progress.append(current_ep_forward)
                    current_ep_return = 0.0
                    current_ep_forward = 0.0
                    obs, _ = self.env.reset()
                    steps_since_reset = 0
                    last_forward_position = None
             
            norm_stats = self.buffer.compute_normalization_stats()
            self.dynamics_model.set_normalization_stats(norm_stats)
                   
            for epoch in range(epochs):
                num = int(self.buffer.current_size)
                shuffled_indices = torch.randperm(num)
                last_loss = 0.0
                for start in range(0, num, batch_size):
                    idx = shuffled_indices[start:start + batch_size]
                    obs_batch, act_batch, next_obs_batch = self.buffer.retrieve_batch(idx)
                    batch_loss = self.dynamics_model.update(obs_batch, act_batch, next_obs_batch)
                    batch_loss_val = float(batch_loss)
                    last_loss = batch_loss_val
                    chunk_loss_sum += batch_loss_val
                    chunk_loss_count += 1
                # last_loss retained for debugging but we report chunk average
            avg_loss_chunk = (chunk_loss_sum / chunk_loss_count) if chunk_loss_count > 0 else float('nan')
            
            episodes_after = len(episode_returns)
            iter_time = time.time() - chunk_start
            episodes_in_chunk = episodes_after - episodes_before
            avg_ep_return_chunk = (sum(episode_returns[-episodes_in_chunk:]) / episodes_in_chunk) if episodes_in_chunk > 0 else float('nan')
            avg_ep_len_chunk = (sum(episode_lengths[-episodes_in_chunk:]) / episodes_in_chunk) if episodes_in_chunk > 0 else float('nan')
            avg_ep_forward_chunk = (sum(episode_forward_progress[-episodes_in_chunk:]) / episodes_in_chunk) if episodes_in_chunk > 0 else float('nan')
            avg_ep_len_display = int(avg_ep_len_chunk) if episodes_in_chunk > 0 else float('nan')
            print(f"Chunk {chunk_index}/{total_chunks}: avg_ep_rew={avg_ep_return_chunk:.2f} | avg_ep_loss={avg_loss_chunk:.5f} | avg_ep_fwd={avg_ep_forward_chunk:.4f} | avg_ep_len={avg_ep_len_display} | episodes={episodes_in_chunk} | time={iter_time:.2f}s")
            reward_history.append(avg_ep_return_chunk)
            loss_history.append(avg_loss_chunk)
            
        elapsed = time.time() - local_start
        print(f"[MPC_ROLLOUT] finished | elapsed={elapsed:.2f}s \n")
        
        
        
    def predict(self, obs):
        import torch
        obs_t = torch.tensor(obs, dtype=torch.float32)
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
        normalization = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in normalization.items()}
        self.dynamics_model.set_normalization_stats(normalization)

        print(f"Loaded dynamics model from {model_path}")
        return self
