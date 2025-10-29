import os
import torch
from algorithms.base_trainer import BaseTrainer

from algorithms.mb_mpc.buffer import ReplayBuffer
from algorithms.mb_mpc.dynamics_model import DynamicsModel
from algorithms.mb_mpc.planner import RandomShootingPlanner


class MBMPCTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        
        self.env = self._make_train_env()
        
        self.buffer = self._make_buffer()
        self.dynamics_model = self._make_dynamics_model()
        self.planner =self._make_planner()
        
    def _make_dynamics_model(self):
        hidden_sizes = self.train_config.get("hidden_sizes", [256, 256])
        learning_rate = float(self.train_config.get("learning_rate", 1e-3))
        
        return DynamicsModel(
            observation_dim = self.env.observation_space.shape[0], 
            action_dim = self.env.action_space.shape[0], 
            hidden_sizes = hidden_sizes, 
            learning_rate = learning_rate)
        
    def _make_buffer(self):        
        init_random_steps = int(self.train_config["init_random_steps"])
        iterations = int(self.train_config["iterations"])
        rollout_steps = int(self.train_config["rollout_steps"])
        buffer_size = init_random_steps + iterations * rollout_steps
        
        return ReplayBuffer(max_size = buffer_size, observation_dim = self.env.observation_space.shape[0], action_dim = self.env.action_space.shape[0])
    
    def _make_planner(self):
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

        return RandomShootingPlanner(
            dynamics_fn=self.dynamics_model.predict_next_state,
            reward_fn=reward_fn,
            horizon=horizon,
            n_candidates=n_candidates,
            act_low=act_low,
            act_high=act_high,
            discount=discount,
            seed=self.train_seed,
        )
        
    def train(self):
        import time
        import numpy as np
        import torch
        
        print("Starting MB-MPC training")
        start_time = time.time()
        train_config = self.train_config

        # === Metric trackers ===
        reward_history = []
        loss_history = []

        # === Config ===
        n_iterations = int(train_config["iterations"])
        epochs = int(train_config["epochs"])
        rollout_steps = int(train_config["rollout_steps"])
        init_random_steps = int(train_config["init_random_steps"])
        batch_size = int(train_config["batch_size"])
        max_path_length = int(train_config.get("max_path_length"))
        
        def _train_one_full_epoch():
            num = int(self.buffer.current_size)
            if num == 0:
                return 0.0
            order = torch.randperm(num)
            last_loss = 0.0
            for start in range(0, num, batch_size):
                idx = order[start:start + batch_size]
                # NOTE: if your ReplayBuffer stores numpy, convert to torch first
                obs_batch       = self.buffer.observations[idx]
                act_batch       = self.buffer.actions[idx]
                next_obs_batch  = self.buffer.next_observations[idx]
                # (Optional) move to model device:
                # obs_batch = obs_batch.to(self.dynamics_model.device); act_batch = ...
                last_loss = self.dynamics_model.update(obs_batch, act_batch, next_obs_batch)
            return float(last_loss)

        # === 1. Random data collection ===
        obs, _ = self.env.reset(seed=self.train_seed)
        
        
        steps_since_reset = 0
        for step in range(init_random_steps):
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.buffer.add(obs, action, next_obs)
            obs = next_obs
            steps_since_reset += 1
            if terminated or truncated or steps_since_reset >= max_path_length:
                obs, _ = self.env.reset()
                steps_since_reset = 0
                
        # We need to do somthing here 
        
        # === Compute and load normalization stats ===
        norm_stats = self.buffer.compute_normalization_stats()
        self.dynamics_model.set_normalization_stats(norm_stats)
        print("DEBUG: obs_std mean =", self.buffer.observations_std.mean().item())
        
        # === 2. Initial model pretraining ===
        for epoch in range(epochs):
            loss = _train_one_full_epoch()
            if (epoch + 1) % 10 == 0:
                print(f"Pretrain Epoch {epoch+1}/{epochs} | Loss: {loss:.5f}")
                
        # === 3. MPC Rollouts ===
        for itr in range(n_iterations):
            total_reward = 0
            obs, _ = self.env.reset()
            iter_start = time.time() 
            
            steps_since_reset = 0

            for step in range(rollout_steps):
                # Plan next action using the learned dynamics
                action = self.planner.plan(torch.tensor(obs, dtype=torch.float32))

                # Convert to NumPy before stepping the environment
                if isinstance(action, torch.Tensor):
                    action = action.detach().cpu().numpy()

                # Step environment with NumPy action
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.buffer.add(obs, action, next_obs)
                total_reward += reward
                obs = next_obs
                
                steps_since_reset += 1

                if terminated or truncated or steps_since_reset >= max_path_length:
                    obs, _ = self.env.reset()
                    steps_since_reset = 0
                
            # === 4. Retrain dynamics model ===
            for epoch in range(epochs):
                loss = _train_one_full_epoch()
 


            # === 5. Track metrics and timing ===
            reward_history.append(total_reward)
            loss_history.append(loss)
            
            iter_time = time.time() - iter_start           # duration for this iteration
            print(f"Iter {itr+1} | reward={total_reward:.2f} | loss={loss:.5f} | iter_time={iter_time:.2f}s")
            # === Refresh normalization stats after each iteration ===
            norm_stats = self.buffer.compute_normalization_stats()
            self.dynamics_model.set_normalization_stats(norm_stats)


        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"Training finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")

        
    def predict(self, obs):
        import torch
        obs_t = torch.tensor(obs, dtype=torch.float32)
        action = self.planner.plan(obs_t)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return action

    def save(self):
        save_path = os.path.join(self.output_dir, "model.pt")
        torch.save({"state_dict": self.dynamics_model.state_dict()}, save_path)
        print(f"Dynamics model saved to {save_path}")

    def load(self, path):
        # Resolve model path
        model_path = path
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")
        # Load the checkpoint
        ckpt = torch.load(model_path, map_location="cpu")
        # Restore model weights
        state_dict = ckpt.get("state_dict", ckpt)
        self.dynamics_model.load_state_dict(state_dict)
        print(f"Loaded dynamics model from {model_path}")
        return self