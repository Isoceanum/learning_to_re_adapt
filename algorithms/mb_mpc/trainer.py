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
        buffer_size = int(self.train_config.get("buffer_size"))
        return ReplayBuffer(
            max_size = buffer_size, 
            observation_dim = self.env.observation_space.shape[0], 
            action_dim = self.env.action_space.shape[0]
        )
    
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

        # === 1. Random data collection ===
                
        # === 1. Random data collection (biased Nagabandi-style) ===
        import torch
        import numpy as np

        obs, _ = self.env.reset(seed=self.train_seed)
        n_steps = init_random_steps
        step = 0
        episode_rewards = []
        ep_rew = 0.0

        cpu_gen = torch.Generator().manual_seed(self.train_seed)
        low = torch.as_tensor(self.env.action_space.low, dtype=torch.float32)
        high = torch.as_tensor(self.env.action_space.high, dtype=torch.float32)

        # Sticky-action parameters
        sticky_horizon = 5           # keep each random action for a few env steps
        current_action = None
        sticky_counter = 0

        # Directional exploration bias (nudges hopper slightly forward)
        bias_strength = 0.15  # between 0.1–0.2 is usually enough
        bias_vector = torch.zeros_like(low)
        bias_vector[0] = bias_strength  # assume first joint/torque contributes to forward push

        while step < n_steps:
            # Sample a new random action every sticky_horizon steps
            if current_action is None or sticky_counter >= sticky_horizon:
                a = torch.rand((1, low.shape[0]), generator=cpu_gen)
                # Add small forward bias to exploration
                biased_a = a + bias_vector
                current_action = (low + biased_a * (high - low)).squeeze(0).numpy()
                sticky_counter = 0

            # Step the environment
            next_obs, reward, terminated, truncated, info = self.env.step(current_action)
            done = terminated or truncated
            self.buffer.add(obs, current_action, next_obs)
            ep_rew += reward
            step += 1
            sticky_counter += 1
            obs = next_obs

            # Reset at end of episode
            if done:
                episode_rewards.append(ep_rew)
                ep_rew = 0.0
                obs, _ = self.env.reset()

        # === Rollout stats for sanity ===
        if episode_rewards:
            mean_ep_rew = np.mean(episode_rewards)
            mean_len = n_steps / len(episode_rewards)
            print(f"[Init] Random phase: {len(episode_rewards)} episodes | mean_ep_rew={mean_ep_rew:.2f} | mean_len={mean_len:.1f}")

        # === 1b. Diversity diagnostics ===
        buf_size = self.buffer.current_size
        print(f"[Init] Buffer filled with {buf_size:,} transitions after {n_steps:,} random steps")

        obs_arr = np.asarray(self.buffer.observations[:buf_size])
        act_arr = np.asarray(self.buffer.actions[:buf_size])
        next_obs_arr = np.asarray(self.buffer.next_observations[:buf_size])

        obs_std = np.std(obs_arr, axis=0).mean()
        act_std = np.std(act_arr, axis=0).mean()
        print(f"[Init] Mean std(obs)={obs_std:.4f} | Mean std(actions)={act_std:.4f}")

        if obs_arr.shape[1] > 0:
            forward_deltas = next_obs_arr[:, 0] - obs_arr[:, 0]
            avg_forward = np.mean(forward_deltas)
            pos_fraction = np.mean(forward_deltas > 0)
            print(f"[Init] Avg Δx = {avg_forward:.4f} | % forward-moving steps = {pos_fraction*100:.1f}%")

        dim_diverse = np.mean(np.std(obs_arr, axis=0) > 1e-3)
        print(f"[Init] Fraction of state dims with std>1e-3: {dim_diverse*100:.1f}%")
        print("=" * 80)

        


        # === Compute and load normalization stats ===
        norm_stats = self.buffer.compute_normalization_stats()
        self.dynamics_model.set_normalization_stats(norm_stats)
        print("DEBUG: obs_std mean =", self.buffer.observations_std.mean().item())

        # === 2. Initial model pretraining ===
        for epoch in range(epochs):
            obs_batch, act_batch, next_obs_batch = self.buffer.sample_batch(batch_size=batch_size)
            loss = self.dynamics_model.update(obs_batch, act_batch, next_obs_batch)
            if (epoch + 1) % 10 == 0:
                print(f"Pretrain Epoch {epoch+1}/{epochs} | Loss: {loss:.5f}")

        # === 3. MPC Rollouts ===
        for itr in range(n_iterations):
            total_reward = 0
            obs, _ = self.env.reset()
            iter_start = time.time() 

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

                if terminated or truncated:
                    break
                
            # === 4. Retrain dynamics model ===
            for epoch in range(epochs):
                obs_batch, act_batch, next_obs_batch = self.buffer.sample_batch(batch_size=batch_size)
                loss = self.dynamics_model.update(obs_batch, act_batch, next_obs_batch)
                
            # Debug: check average predicted forward delta (Δx)
            with torch.no_grad():
                preds = self.dynamics_model.predict_next_state(obs_batch, act_batch)
                delta_x = (preds[:, 0] - obs_batch[:, 0]).mean().item()
                print(f"DEBUG Δx_pred_mean: {delta_x:.5f}")

            # === 5. Track metrics and timing ===
            reward_history.append(total_reward)
            loss_history.append(loss)
            
            iter_time = time.time() - iter_start           # duration for this iteration
            print(f"Iter {itr+1} | reward={total_reward:.2f} | loss={loss:.5f} | iter_time={iter_time:.2f}s")
            # === Refresh normalization stats after each iteration ===
            norm_stats = self.buffer.compute_normalization_stats()
            print("DEBUG true Δx_mean:", (self.buffer.next_observations[:, 0] - self.buffer.observations[:, 0]).mean().item())
            self.dynamics_model.set_normalization_stats(norm_stats)
            print("actions_std mean:", self.buffer.actions_std.mean().item())


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
        
        torch.save({
            "state_dict": self.dynamics_model.state_dict(),
            "norm_stats": {
                "observations_mean": self.dynamics_model.observations_mean,
                "observations_std": self.dynamics_model.observations_std,
                "actions_mean": self.dynamics_model.actions_mean,
                "actions_std": self.dynamics_model.actions_std,
                "delta_mean": self.dynamics_model.delta_mean,
                "delta_std": self.dynamics_model.delta_std,
            },
        }, save_path)
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
        
        if "norm_stats" in ckpt:
            self.dynamics_model.set_normalization_stats(ckpt["norm_stats"])
            print("Loaded normalization stats into dynamics model.")
            
        else:
            raise ValueError("Missing normalization stats in checkpoint.")
        
        print(f"Loaded dynamics model from {model_path}")
        return self
