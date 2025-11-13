import os
import torch
from algorithms.base_trainer import BaseTrainer
from algorithms.grbal_lite.buffer import ReplayBuffer
from algorithms.grbal_lite.dynamics_model import DynamicsModel
from algorithms.grbal_lite.planner import RandomShootingPlanner
from algorithms.grbal_lite.meta_trainer import MetaTrainer
import time
import numpy as np

from algorithms.mb_mpc.planner import CrossEntropyMethodPlanner
from utils.seed import set_seed

class GrBALLiteTrainer(BaseTrainer):
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
            )
              
        raise AttributeError(f"Planner type {planner_type} not supported")
        
    
    def _make_meta_trainer(self):    
        past_length = int(self.train_config.get("recent_window_size"))
        future_length = int(self.train_config.get("meta_future_length"))
        batch_size = int(self.train_config.get("meta_batch_size"))
        inner_lr = float(self.train_config.get("inner_lr"))
        inner_steps = int(self.train_config.get("inner_steps"))
        first_order = bool(self.train_config.get("meta_first_order"))
        outer_lr = float(self.train_config.get("meta_outer_lr"))
        
        return MetaTrainer(
            self.dynamics_model,
            self.buffer,
            past_length,
            future_length,
            batch_size,
            inner_lr,
            inner_steps,
            first_order,
            outer_lr,
        )
           
    def _plan_after_inner_update(self, obs, buffer):
        recent_window_size = int(self.train_config.get("recent_window_size"))
        inner_steps = int(self.train_config.get("inner_steps"))
        inner_lr = float(self.train_config.get("inner_lr"))
        
        # If buffer holds insufficient transitions, we skip inner loop 
        if buffer.episode_size() < recent_window_size:
            return self.planner.plan(torch.as_tensor(obs, dtype=torch.float32))
        
        # Save dynamics_model_snapshot for later restoration
        state = self.dynamics_model.state_dict()
        dynamics_model_snapshot = {}
        for k, v in state.items():
            dynamics_model_snapshot[k] = v.detach().clone()
            
        # Retrieve n last transitions, n = recent_window_size
        observations, actions, next_observations = buffer.retrieve_recent_transitions_in_episode(recent_window_size)
        
        # perform n stochastic gradient descent steps on loss from recent transitions, n = inner_steps
        for _ in range(inner_steps):
            # compute loss between predicted_next_observations and real next_observations
            loss = self.dynamics_model.compute_normalized_delta_loss(observations, actions, next_observations)

            # clear existing gradients
            self.dynamics_model.zero_grad(set_to_none=True)
            # compute gradients for all model params
            loss.backward()
            with torch.no_grad():
                for p in self.dynamics_model.parameters():
                    if p.grad is not None:
                        p -= inner_lr * p.grad
                        
        action = self.planner.plan(torch.as_tensor(obs, dtype=torch.float32))
        
        # restore dynamics_model to snapshot
        self.dynamics_model.load_state_dict(dynamics_model_snapshot)
        
        return action
        
    def _collect_warmup_data(self):
        local_start = time.time()
        total_env_steps = int(self.train_config["total_env_steps"])
        warmup_steps_budget = int(self.train_config["warmup_steps_budget"])
        max_path_length = int(self.train_config["max_path_length"])
       
        print(f"[COLLECT_WARMUP_DATA] warmup_steps_budget={warmup_steps_budget} | seed={self.train_seed}")
        
        obs, _ = self.env.reset(seed=self.train_seed)
        self.buffer.set_episode_start()

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
                self.buffer.set_episode_start()
                reset_counter += 1
                steps_since_reset = 0
        
        # === Compute and load normalization stats ===
        norm_stats = self.buffer.compute_normalization_stats()
        self.dynamics_model.set_normalization_stats(norm_stats)
        
        elapsed = time.time() - local_start
        buffer_size = int(self.buffer.current_size)
        self.buffer.set_warmup_end_index()
        print(f"[COLLECT_WARMUP_DATA] episodes={reset_counter} | buffer_size={buffer_size} | elapsed={elapsed:.2f}s \n")
    
    def _pretrain_dynamics_model(self):
        local_start = time.time()
        epochs = int(self.train_config["epochs"])
        batch_size = int(self.train_config["batch_size"])
        
        print(f"[PRETRAIN_DYNAMICS] epochs={epochs} | batch_size={batch_size}")

        for epoch in range(epochs):
            epoch_seed = self.train_seed + epoch
            generator = torch.Generator().manual_seed(epoch_seed)
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
                
    def _run_mpc_rollout(self):
        local_start = time.time()
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
        
        meta_enabled = bool(self.train_config.get("meta_enabled"))
        meta_trainer = None
        
        if meta_enabled:
            meta_trainer = self._make_meta_trainer()
            print("[MPC_ROLLOUT][meta] one outer step per chunk")
        
        obs, _ = self.env.reset()
        self.buffer.set_episode_start()

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

        
            for _ in range(steps_this_chunk):
                action = self._plan_after_inner_update(obs, self.buffer)
                
                if isinstance(action, torch.Tensor):
                    action = action.detach().cpu().numpy()
                
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                forward_pos = self._get_forward_position(info)
                
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
                    self.buffer.set_episode_start()

                    steps_since_reset = 0
                    last_forward_position = None
            
            if meta_enabled:
                avg_loss_chunk = float("nan")  # θ* updates come solely from the meta gradient
            else:
                avg_loss_chunk = self._train_on_buffer(epochs, batch_size)
            min_transitions_for_meta = int(self.train_config.get("recent_window_size")) + int(self.train_config.get("meta_future_length"))
            
            if meta_enabled and (self.buffer.current_size >= min_transitions_for_meta):
                normalization_stats = self.buffer.compute_normalization_stats()
                self.dynamics_model.set_normalization_stats(normalization_stats)
                support_loss_val, query_loss_val = meta_trainer.run_outer_iteration()
                
            outer_loss_diff = support_loss_val - query_loss_val
            episodes_after = len(episode_returns)
            iter_time = time.time() - chunk_start
            episodes_in_chunk = episodes_after - episodes_before
            avg_ep_return_chunk = (sum(episode_returns[-episodes_in_chunk:]) / episodes_in_chunk) if episodes_in_chunk > 0 else float('nan')
            avg_ep_len_chunk = (sum(episode_lengths[-episodes_in_chunk:]) / episodes_in_chunk) if episodes_in_chunk > 0 else float('nan')
            avg_ep_forward_chunk = (sum(episode_forward_progress[-episodes_in_chunk:]) / episodes_in_chunk) if episodes_in_chunk > 0 else float('nan')
            avg_ep_len_display = int(avg_ep_len_chunk) if episodes_in_chunk > 0 else float('nan')
            print(f"Chunk {chunk_index}/{total_chunks}: avg_ep_rew={avg_ep_return_chunk:.2f} | outer_loss_diff={outer_loss_diff:.5f} | avg_ep_fwd={avg_ep_forward_chunk:.4f} | avg_ep_len={avg_ep_len_display} | episodes={episodes_in_chunk} | time={iter_time:.2f}s")
            reward_history.append(avg_ep_return_chunk)
            loss_history.append(avg_loss_chunk)
            
        elapsed = time.time() - local_start
        print(f"[MPC_ROLLOUT] finished | elapsed={elapsed:.2f}s \n")
           
    def _train_on_buffer(self, epochs, batch_size):
        norm_stats = self.buffer.compute_normalization_stats()
        self.dynamics_model.set_normalization_stats(norm_stats)
        
        chunk_loss_sum = 0.0
        chunk_loss_count = 0
                
        for epoch in range(epochs):
            num = int(self.buffer.current_size)
            
            if num == 0:
                return 0.0
            
            shuffled_indices = torch.randperm(num)
            
            for start in range(0, num, batch_size):
                idx = shuffled_indices[start:start + batch_size]
                obs_batch, act_batch, next_obs_batch = self.buffer.retrieve_batch(idx)
                batch_loss = self.dynamics_model.update(obs_batch, act_batch, next_obs_batch)
                batch_loss_val = float(batch_loss)
                chunk_loss_sum += batch_loss_val
                chunk_loss_count += 1
        avg_loss_chunk = (chunk_loss_sum / chunk_loss_count) if chunk_loss_count > 0 else float('nan')
        
        return avg_loss_chunk

    def train(self):
        print("Starting GrBal training")
        start_time = time.time()

        # === 1. Random data collection ===
        self._collect_warmup_data()

        # === 2. Initial model pretraining ===
        self._pretrain_dynamics_model()
        
        # === 3. MPC Rollouts ===
        self._run_mpc_rollout()
        
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

    def evaluate(self):
        print("[EVAL] GrBAL-lite override")
        
        max_path_length = int(self.train_config.get("max_path_length"))
        if int(self.train_config["recent_window_size"]) > max_path_length: print("[EVAL][warn] recent_window_size exceeds max_path_length — inner adaptation will never trigger.")
        buffer_size = max_path_length
        eval_buffer = ReplayBuffer(max_size=buffer_size, observation_dim=self.env.observation_space.shape[0], action_dim=self.env.action_space.shape[0])
        
        episodes = int(self.eval_config["episodes"])
        seeds = self.eval_config["seeds"]
        
        all_rewards = []
        forward_progresses = []
        episode_lengths = []    
        total_runs = len(seeds) * episodes

        print(f"Evaluating {episodes} episode(s) × {len(seeds)} seed(s) = {total_runs} total runs")

        eval_start_time = time.time()

        for seed in seeds:
            set_seed(seed)
            eval_env = self._make_eval_env(seed=seed)
            
            seed_rewards = []
            seed_forward = []
            seed_lengths = []
            
            for episode in range(episodes):
                obs, _ = eval_env.reset()
                
                eval_buffer.clear()

                com_x_start = None

                done = False
                ep_reward = 0.0
                steps = 0
                last_com_x = None

                while not done:
                    action = self._plan_after_inner_update(obs, eval_buffer)
                    
                    if isinstance(action, torch.Tensor): 
                        action = action.detach().cpu().numpy()
                    

                    next_obs, reward, terminated, truncated, info = eval_env.step(action)
                    eval_buffer.add(obs, action, next_obs)
                    obs = next_obs
                    
                    
                    done = terminated or truncated
                    ep_reward += float(reward)
                    steps += 1
                    
                    if com_x_start is None:
                        com_x_start = self._get_forward_position(info)
                    last_com_x = self._get_forward_position(info)
                    
                    if steps >= max_path_length: 
                        done = True


                # Compute forward progress
                fp = last_com_x - com_x_start if (com_x_start is not None and last_com_x is not None) else 0.0
                
                
                seed_rewards.append(ep_reward)
                seed_forward.append(fp)
                seed_lengths.append(steps)

                all_rewards.append(ep_reward)
                forward_progresses.append(fp)
                episode_lengths.append(steps)
                eval_env.close()

            print(f"Seed {seed}: reward={np.mean(seed_rewards):.1f} ± {np.std(seed_rewards):.1f}, "f"fp={np.mean(seed_forward):.2f} ± {np.std(seed_forward):.1f}")
            
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        fp_mean = np.mean(forward_progresses)
        fp_std = np.std(forward_progresses)
        ep_len_mean = np.mean(episode_lengths)
        elapsed = time.time() - eval_start_time
        elapsed_str = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"

        print("\nEvaluation summary:")
        print(f"- reward_mean: {mean_reward:.4f}")
        print(f"- reward_std: {std_reward:.4f}")
        print(f"- forward_progress_mean: {fp_mean:.4f}")
        print(f"- forward_progress_std: {fp_std:.4f}")
        print(f"- episode_length_mean: {ep_len_mean:.2f}")
        print(f"- elapsed: {elapsed_str}")
