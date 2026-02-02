import copy
import os
import yaml
from algorithms.base_trainer import BaseTrainer
import numpy as np
import torch
import torch.optim as optim
import time
from utils.seed import set_seed
from algorithms.untils import make_dynamics_model, make_planner
from evaluation.model_error import compute_k_step_rmse_for_episode, compute_top_rmse_by_dim_for_episode
from algorithms.tsra.residual_adapter import ResidualAdapter
from algorithms.tsra.normalizer import Normalizer
from algorithms.tsra.residual_dynamics_wrapper import ResidualDynamicsWrapper
from torch.utils.data import DataLoader, TensorDataset


class TSRATrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        self.env = self._make_train_env() # make the env used for training
        self.pretrained_dynamics_model = self.load_pretrained_dynamics_model() # load a pretrained dynamics model
        self.residual_adapter = self._make_residual_adapter() # make residual adapter
        self.optimizer = self._make_optimizer() # make optimizer
        self.normalizer = self._make_normalizer()
        self.residual_dynamics_wrapper = self._make_residual_dynamics_wrapper() # make base + residual adapter wrapper 
        self.planner = self._make_planner() # make planner 
        
        
    def _make_normalizer(self):
        if self.residual_adapter is None:
            return 
        base = self.pretrained_dynamics_model
        device = next(self.residual_adapter.parameters()).device
        
        mean_obs = base.mean_obs.to(device)
        std_obs  = base.std_obs.to(device)
        mean_act = base.mean_act.to(device)
        std_act  = base.std_act.to(device)
        
        obs_dim = self.env.observation_space.shape[0]
        mean_residual = torch.zeros(obs_dim, device=device)
        std_residual  = torch.ones(obs_dim, device=device)

        return Normalizer(mean_obs, std_obs, mean_act, std_act, mean_residual, std_residual)
        
    def _make_residual_adapter(self):   
        residual_adapter_config = self.train_config.get("residual_adapter")
        if residual_adapter_config is None or not residual_adapter_config.get("enabled", False):
            return None
        
        hidden_sizes = residual_adapter_config.get("hidden_sizes")
        return ResidualAdapter(self.env.observation_space.shape[0], self.env.action_space.shape[0], hidden_sizes).to(self.device)
        
    def _make_optimizer(self):
        if self.residual_adapter is None:
            return
        learning_rate = float(self.train_config["residual_adapter"]["learning_rate"])
        return torch.optim.AdamW(self.residual_adapter.parameters(), lr=learning_rate, weight_decay=1e-4)

    def _make_residual_dynamics_wrapper(self):
        return ResidualDynamicsWrapper(self.pretrained_dynamics_model, self.residual_adapter, self.normalizer) 
        
     
    def _make_planner(self):
        planner_config = self.train_config.get("planner")
        
        dynamics_fn = self.pretrained_dynamics_model.predict_next_state if self.residual_adapter is None else self.residual_dynamics_wrapper.predict_next_state    
        action_space = self.env.action_space
        
        base_env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(base_env, "get_model_reward_fn"): 
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")
        
        reward_fn = base_env.get_model_reward_fn()
        return make_planner(planner_config, dynamics_fn, reward_fn, action_space, self.device, self.train_seed)
    
    def load_pretrained_dynamics_model(self):
        model_path = self.train_config["pretrained_dynamics_model"]["model_path"]
        config_path = self.train_config["pretrained_dynamics_model"]["config_path"]
        
        with open(config_path, "r") as f:
            pretrained_dynamics_model_config = yaml.safe_load(f)
                        
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        pretrained_dynamics_model = make_dynamics_model(pretrained_dynamics_model_config, obs_dim, action_dim, self.train_seed).to(self.device)
        pretrained_dynamics_model.load_saved_model(model_path)
        pretrained_dynamics_model.freeze()
        return pretrained_dynamics_model
    
    
    
    def _collect_env_steps(self, iteration_index, steps_target, max_episode_length):
        steps_collected_this_iteration = 0
            
        collect_start_time = time.time()
        
        obs_all = []
        act_all = []
        next_obs_all = []
        
        while steps_collected_this_iteration < steps_target:
            obs, _ = self.env.reset()

            episode_steps = 0
            while episode_steps < max_episode_length:
                
                if iteration_index == 0:
                    _base_planner = copy.copy(self.planner) 
                    _base_planner.dynamics_fn = self.pretrained_dynamics_model.predict_next_state
                    action = _base_planner.plan(obs)
                    
                else:
                    action = self.planner.plan(obs)
                    
                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy()
                    
                next_obs, reward, terminated, truncated, info = self._step_env(action)  
                obs_all.append(obs)
                act_all.append(action)
                next_obs_all.append(next_obs)
                obs = next_obs
            
                episode_steps += 1
                steps_collected_this_iteration += 1
                            
                if steps_collected_this_iteration >= steps_target or terminated or truncated:
                    break
                
        steps_collected_this_iteration = steps_collected_this_iteration
        log_collect_time = time.time() - collect_start_time
                
        print(f"collect: " f"steps={steps_collected_this_iteration} " f"time={log_collect_time:.1f}s")
        return obs_all, act_all, next_obs_all
                

    def _split_data(self, obs_all, act_all, next_obs_all):
        valid_split_ratio = float(self.train_config["valid_split_ratio"])
        
        obs_all = np.asarray(obs_all, dtype=np.float32)
        act_all = np.asarray(act_all, dtype=np.float32)
        next_obs_all = np.asarray(next_obs_all, dtype=np.float32)
        
        
        n = obs_all.shape[0]
        idx = np.random.permutation(n)
        
        n_val = int(n * valid_split_ratio)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        
        train_obs = obs_all[train_idx]
        train_act = act_all[train_idx]
        train_next_obs = next_obs_all[train_idx]

        val_obs = obs_all[val_idx]
        val_act = act_all[val_idx]
        val_next_obs = next_obs_all[val_idx]
        
        train_obs_t = torch.as_tensor(train_obs, dtype=torch.float32, device=self.device)
        train_act_t = torch.as_tensor(train_act, dtype=torch.float32, device=self.device)
        train_next_obs_t = torch.as_tensor(train_next_obs, dtype=torch.float32, device=self.device)

        val_obs_t = torch.as_tensor(val_obs, dtype=torch.float32, device=self.device)
        val_act_t = torch.as_tensor(val_act, dtype=torch.float32, device=self.device)
        val_next_obs_t = torch.as_tensor(val_next_obs, dtype=torch.float32, device=self.device)
        
        train_ds = TensorDataset(train_obs_t, train_act_t, train_next_obs_t)
        val_ds = TensorDataset(val_obs_t, val_act_t, val_next_obs_t)

        batch_size = int(self.train_config["batch_size"])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        return train_loader, val_loader


    def _update_residual_stats_from_loader(self, train_loader, iteration_index):
        if iteration_index > 2:
            return 
        
        # If you've frozen the normalizer, this will just return (per your Normalizer code)
        residual_chunks = []

        with torch.no_grad():
            for obs, act, next_obs in train_loader:
                # obs/act/next_obs are already torch tensors on self.device (because you moved them in _split_data)

                base_pred_next = self.pretrained_dynamics_model.predict_next_state(obs, act)

                true_delta = next_obs - obs
                base_delta = base_pred_next - obs
                residual_raw = true_delta - base_delta  # [B, obs_dim]

                residual_chunks.append(residual_raw)

        residual_all = torch.cat(residual_chunks, dim=0)  # [N_train, obs_dim]
        self.normalizer.update_residual_stats_from_raw(residual_all)

    def train(self):
        print("Starting Task Specific Task Residual Adapter training")           
        start_time = time.time()
        
        if self.residual_adapter is None:
            print("No residual_adapter specified, training skipped" )
            return
        
        
        max_episode_length = int(self.train_config["max_episode_length"])
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        iterations = int(self.train_config["iterations"])
        
        
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        train_epochs = int(self.train_config["train_epochs"])
        
        
        penalty_lambda = 0.0
        for iteration_index in range(iterations):
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            obs_all, act_all, next_obs_all = self._collect_env_steps(iteration_index, steps_per_iteration, max_episode_length)
            train_loader, val_loader = self._split_data(obs_all, act_all, next_obs_all)
            self._update_residual_stats_from_loader(train_loader, iteration_index)
            
            for epoch in range(train_epochs):
                
                train_loss_sum = 0.0
                train_batches = 0
                self.residual_adapter.train()
                for obs, act, next_obs in train_loader:
                    with torch.no_grad():
                        base_pred_next = self.pretrained_dynamics_model.predict_next_state(obs, act)
                        
                    true_delta = next_obs - obs
                    base_delta = base_pred_next - obs
                    residual_raw = true_delta - base_delta
                    
                    obs_norm = self.normalizer.normalize_observations(obs)
                    act_norm = self.normalizer.normalize_actions(act)
                    residual_target_norm = self.normalizer.normalize_residual(residual_raw)
                    
                    pred_residual_norm = self.residual_adapter(obs_norm, act_norm)
                    
                    mse = torch.mean((pred_residual_norm - residual_target_norm) ** 2)
                    penalty = torch.mean(pred_residual_norm ** 2)
                    loss = mse + (penalty_lambda * penalty)
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                    train_loss_sum += float(loss.item())
                    train_batches += 1
                    
                    
                avg_train_loss = train_loss_sum / max(1, train_batches)
                #print(f"iter {iteration_index} epoch {epoch}: train_loss={avg_train_loss:.6f}")
                
                
                # --- val ---
                val_loss_sum = 0.0
                val_batches = 0
                self.residual_adapter.eval()
                with torch.no_grad():
                    for obs, act, next_obs in val_loader:
                        base_pred_next = self.pretrained_dynamics_model.predict_next_state(obs, act)

                        true_delta = next_obs - obs
                        base_delta = base_pred_next - obs
                        residual_raw = true_delta - base_delta

                        obs_norm = self.normalizer.normalize_observations(obs)
                        act_norm = self.normalizer.normalize_actions(act)
                        residual_target_norm = self.normalizer.normalize_residual(residual_raw)

                        pred_residual_norm = self.residual_adapter(obs_norm, act_norm)

                        mse = torch.mean((pred_residual_norm - residual_target_norm) ** 2)
                        penalty = torch.mean(pred_residual_norm ** 2)
                        loss = mse + penalty_lambda * penalty

                        val_loss_sum += float(loss.item())
                        val_batches += 1

                avg_val_loss = val_loss_sum / max(1, val_batches)

                print(f"iter {iteration_index} epoch {epoch}: train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f}")
                
                
                
            obs_b, act_b, _ = next(iter(train_loader))
            obs_b = obs_b.to(self.device)
            act_b = act_b.to(self.device)

            self.residual_adapter.eval()
            self.residual_dynamics_wrapper.print_correction_norm_ratio(
                obs_b, act_b, prefix=f"[iter {iteration_index} post] "
            )
                
                
            
                        
                        
            
        
        
        elapsed = int(time.time() - start_time)
        print(f"\nTraining finished. Elapsed: {elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}")


    def evaluate(self):
        print("Overwriting base evaluate to predict model error")
        seeds = self.eval_config["seeds"]
        k_list = self.eval_config["k_list"]
        max_episode_length = int(self.train_config["max_episode_length"])
        max_k = max(k_list)
        
        eval_start_time = time.time()
        
        episode_rewards = []
        episode_forward_progresses = []

        # --- per-model accumulators (BASE vs BASE+RA) ---
        base_rmse_values_by_k = {k: [] for k in k_list}
        ra_rmse_values_by_k = {k: [] for k in k_list}

        base_top_dim_counts_k1 = {}
        ra_top_dim_counts_k1 = {}

        for seed in seeds:
            set_seed(seed)
            env = self._make_eval_env(seed=seed)

            obs, _ = env.reset(seed=seed)
            done = False
            steps = 0
            episode_transitions = []
            ep_reward = 0.0
            com_x_start = None
            last_com_x = None

            while not done and steps < max_episode_length:
                action = self.predict(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                episode_transitions.append((obs, action, next_obs))
                obs = next_obs
                ep_reward += float(reward)
                done = terminated or truncated
                steps += 1

                # assumes x_position exists in info
                if com_x_start is None:
                    com_x_start = float(info["x_position"])
                last_com_x = float(info["x_position"])

            forward_progress = (
                last_com_x - com_x_start
                if (com_x_start is not None and last_com_x is not None)
                else 0.0
            )
            
            episode_rewards.append(ep_reward)
            episode_forward_progresses.append(forward_progress)
            env.close()
            
            print(f"\n------------------[seed {seed}]------------------")
            print(f"[rollout] reward={ep_reward:.4f} forward_progress={forward_progress:.4f} len={steps}")
            
            # ---------------- K-step RMSE ----------------
            print()
            base_rmse_by_k = compute_k_step_rmse_for_episode(episode_transitions, self.pretrained_dynamics_model, k_list, self.device)
            for k in k_list:
                base_rmse_values_by_k[k].append(base_rmse_by_k[k])

            print("[BASE]    RMSE:", " | ".join([f"k-{k} {base_rmse_by_k[k]:.4f}" for k in k_list]))
            
            ra_rmse_by_k = compute_k_step_rmse_for_episode(episode_transitions, self.residual_dynamics_wrapper, k_list, self.device)
            for k in k_list:
                ra_rmse_values_by_k[k].append(ra_rmse_by_k[k])

            print("[BASE+RA] RMSE:", " | ".join([f"k-{k} {ra_rmse_by_k[k]:.4f}" for k in k_list]))
            
            # ---------------- Top dim by rmse error 1 step ----------------
            print()
            base_top_by_k = compute_top_rmse_by_dim_for_episode(episode_transitions, self.pretrained_dynamics_model, max_k, self.device, 5)
            for idx, _ in base_top_by_k[1]:
                base_top_dim_counts_k1[idx] = base_top_dim_counts_k1.get(idx, 0) + 1

            print("[BASE]    Top dims k-1 :", " | ".join([f"({idx}):{val:.4f}" for idx, val in base_top_by_k[1]]))
            
            ra_top_by_k = compute_top_rmse_by_dim_for_episode(episode_transitions, self.residual_dynamics_wrapper, max_k, self.device, 5)
            for idx, _ in ra_top_by_k[1]:
                ra_top_dim_counts_k1[idx] = ra_top_dim_counts_k1.get(idx, 0) + 1

            print("[BASE+RA] Top dims k-1 :", " | ".join([f"({idx}):{val:.4f}" for idx, val in ra_top_by_k[1]]))
            
            # ---------------- Top dim by rmse error max k step ----------------
            print()
            print(f"[BASE]    Top dims k-{max_k}:", " | ".join([f"({idx}):{val:.4f}" for idx, val in base_top_by_k[max_k]]))
            print(f"[BASE+RA] Top dims k-{max_k}:", " | ".join([f"({idx}):{val:.4f}" for idx, val in ra_top_by_k[max_k]]))

        # ---------------- summary ----------------
        print("\n--------------------")

        base_mean_rmse_by_k = {k: float(np.mean(base_rmse_values_by_k[k])) for k in k_list}
        print("[BASE]    RMSE mean:", " | ".join([f"k-{k} {base_mean_rmse_by_k[k]:.4f}" for k in k_list]))
        ra_mean_rmse_by_k = {k: float(np.mean(ra_rmse_values_by_k[k])) for k in k_list}
        print("[BASE+RA] RMSE mean:", " | ".join([f"k-{k} {ra_mean_rmse_by_k[k]:.4f}" for k in k_list]))
        
        print()
        
        base_top_dims_sorted = sorted(base_top_dim_counts_k1.items(), key=lambda kv: kv[1], reverse=True)
        print("[BASE]    top_dims_k1_freq:", " | ".join([f"({idx})x{cnt}" for idx, cnt in base_top_dims_sorted[:10]]))
        ra_top_dims_sorted = sorted(ra_top_dim_counts_k1.items(), key=lambda kv: kv[1], reverse=True)
        print("[BASE+RA] top_dims_k1_freq:", " | ".join([f"({idx})x{cnt}" for idx, cnt in ra_top_dims_sorted[:10]]))
        
        print("\n[summary]")
        reward_mean = float(np.mean(episode_rewards))
        reward_std = float(np.std(episode_rewards))
        print(f"- reward: {reward_mean:.4f} ± {reward_std:.4f}")
        
        fp_mean = float(np.mean(episode_forward_progresses))
        fp_std = float(np.std(episode_forward_progresses))
        print(f"- forward_progress: {fp_mean:.4f} ± {fp_std:.4f}")
        
        elapsed = time.time() - eval_start_time
        elapsed_str = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
        print(f"- elapsed: {elapsed_str}")
            
    def predict(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = self.planner.plan(obs_t)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return action
       
    def save(self):
        print("no saving for now")
   
