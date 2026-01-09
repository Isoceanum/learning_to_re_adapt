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
from algorithms.humble_residual_adapter.transition_buffer import TransitionBuffer
from algorithms.humble_residual_adapter.residual_adapter import ResidualAdapter
from algorithms.humble_residual_adapter.residual_dynamics_wrapper import ResidualDynamicsWrapper

class HumbleResidualAdapterTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        self.env = self._make_train_env()
        
        self.pretrained_dynamics_model = self.load_pretrained_dynamics_model()
        self.residual_adapter = self._make_residual_adapter()
        self.optimizer = self._make_optimizer()
        self.residual_dynamics_wrapper = self._make_residual_dynamics_wrapper()
        self.planner = self._make_planner()
        self.buffer = self._make_buffer()
        
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
        return ResidualDynamicsWrapper(self.pretrained_dynamics_model, self.residual_adapter) 
        
    def _make_buffer(self):
        valid_split_ratio = float(self.train_config["valid_split_ratio"])
        return TransitionBuffer(valid_split_ratio, self.train_seed)
     
    def _make_planner(self):
        planner_config = self.train_config.get("planner")
        
        residual_adapter_config = self.train_config.get("residual_adapter")
        dynamics_fn = self.pretrained_dynamics_model.predict_next_state if residual_adapter_config is None else self.residual_dynamics_wrapper.predict_next_state    
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
        
        pretrained_dynamics_model = make_dynamics_model(pretrained_dynamics_model_config, obs_dim, action_dim, self.train_seed)
        pretrained_dynamics_model.load_saved_model(model_path)
        pretrained_dynamics_model.freeze()
        return pretrained_dynamics_model
    
    def _collect_env_steps(self, policy, steps_target, max_episode_length):
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
            episode_base_pred_next = []
            
            while episode_steps < max_episode_length:
                if policy == "planner":
                    action = self.planner.plan(obs)
                    if torch.is_tensor(action):
                        action = action.detach().cpu().numpy()
                else:
                    action = self.env.action_space.sample()
                
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
                    base_pred_next = self.pretrained_dynamics_model.predict_next_state(obs_t, act_t)

                episode_base_pred_next.append(base_pred_next.squeeze(0).cpu().numpy())
                    
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
                
            self.buffer.add_trajectory(episode_obs, episode_act, episode_next_obs, episode_base_pred_next)
            log_episode_forward_progress.append(float(episode_x_last - episode_x_start))
            log_episode_velocity.append(float(episode_velocity))
            log_episode_returns.append(float(episode_return))
            
        reward_mean = float(np.mean(log_episode_returns))
        reward_std = float(np.std(log_episode_returns))
                    
        num_train_transitions = sum(len(ep) for ep in self.buffer.train_observations)
        avg_forward_progress = sum(log_episode_forward_progress) / max(1, len(log_episode_forward_progress))
        avg_velocity = sum(log_episode_velocity) / max(1, len(log_episode_velocity))
        steps_collected_this_iteration = steps_collected_this_iteration
        log_collect_time = time.time() - log_collect_start_time
        log_episodes = log_episodes
        
        print(f"collect: dataset={num_train_transitions} " f"steps={steps_collected_this_iteration} " f"episodes={log_episodes} " f"reward_mean={reward_mean:.3f}  ± {reward_std:.3f}  " f"avg_fp={avg_forward_progress:.3f} " f"avg_v={avg_velocity:.3f} " f"time={log_collect_time:.1f}s")
                
    def _print_iteration_k_step_rmse(self):
        k_list=(1, 2, 5, 10, 15)

        eval_obs_eps = self.buffer.eval_observations
        eval_act_eps = self.buffer.eval_actions
        eval_next_obs_eps = self.buffer.eval_next_observations

        base_vals = {k: [] for k in k_list}
        ra_vals = {k: [] for k in k_list}

        for i in range(len(eval_obs_eps)):
            episode_transitions = list(zip(eval_obs_eps[i], eval_act_eps[i], eval_next_obs_eps[i]))

            base_rmse_by_k = compute_k_step_rmse_for_episode(episode_transitions, self.pretrained_dynamics_model, k_list, self.device)
            ra_rmse_by_k = compute_k_step_rmse_for_episode(episode_transitions, self.residual_dynamics_wrapper, k_list, self.device)

            for k in k_list:
                base_vals[k].append(base_rmse_by_k[k])
                ra_vals[k].append(ra_rmse_by_k[k])

        base_mean = {k: float(np.mean(base_vals[k])) for k in k_list}
        ra_mean = {k: float(np.mean(ra_vals[k])) for k in k_list}
        
        print()
        print("[BASE]    RMSE mean:", " | ".join([f"k-{k} {base_mean[k]:.4f}" for k in k_list]))
        print("[BASE+RA] RMSE mean:", " | ".join([f"k-{k} {ra_mean[k]:.4f}" for k in k_list]))
        
    def train_for_iteration(self, train_epochs, batch_size):
        print_every_n_epochs = 5
        
        
        for epoch in range(train_epochs):
            epoch_SDG_steps = 0
            num_train_transitions = sum(len(ep) for ep in self.buffer.train_observations)
            updates_per_epoch = max(1, int(np.ceil(num_train_transitions / batch_size)))
            epoch_losses = []

            for _ in range(updates_per_epoch):
                obs_b, act_b, next_obs_b, base_pred_next_b = self.buffer.sample_transitions(batch_size, split="train")
                obs_b = obs_b.to(self.device)
                act_b = act_b.to(self.device)
                next_obs_b = next_obs_b.to(self.device)
                base_pred_next_b = base_pred_next_b.to(self.device)
                
                # --- derive deltas ---
                base_pred_delta = base_pred_next_b - obs_b
                true_delta = next_obs_b - obs_b
                r_delta_target = true_delta - base_pred_delta

                # --- normalize target residual ---
                r_delta_target_norm = (r_delta_target - self.residual_adapter.residual_mean) / self.residual_adapter.residual_std

                # --- forward adapter (predict normalized residual correction) ---
                delta_correction_norm = self.residual_adapter(obs_b, act_b, base_pred_delta)

                # --- loss + step ---
                loss = torch.mean((delta_correction_norm - r_delta_target_norm) ** 2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_SDG_steps += 1
                epoch_losses.append(loss.item())
                
                            
            should_print_epoch_summery = (epoch % print_every_n_epochs == 0) or (epoch == train_epochs - 1)
            
            if should_print_epoch_summery:
                num_eval_transitions = sum(len(ep) for ep in self.buffer.eval_observations)
                eval_updates_per_epoch = max(1, int(np.ceil(num_eval_transitions / batch_size)))
                epoch_eval_losses = []

                with torch.no_grad():
                    for _ in range(eval_updates_per_epoch):
                        obs_b, act_b, next_obs_b, base_pred_next_b = self.buffer.sample_transitions(batch_size, split="eval")
                        obs_b = obs_b.to(self.device)
                        act_b = act_b.to(self.device)
                        next_obs_b = next_obs_b.to(self.device)
                        base_pred_next_b = base_pred_next_b.to(self.device)

                        base_pred_delta = base_pred_next_b - obs_b
                        true_delta = next_obs_b - obs_b
                        r_delta_target = true_delta - base_pred_delta

                        r_delta_target_norm = (r_delta_target - self.residual_adapter.residual_mean) / self.residual_adapter.residual_std
                        delta_correction_norm = self.residual_adapter(obs_b, act_b, base_pred_delta)

                        eval_loss = torch.mean((delta_correction_norm - r_delta_target_norm) ** 2)
                        epoch_eval_losses.append(eval_loss.item())
                                
                eval_mean = np.mean(epoch_eval_losses) if len(epoch_eval_losses) > 0 else float("nan")
                print(f"epoch {epoch+1}/{train_epochs}: train={np.mean(epoch_losses):.6f} eval={eval_mean:.6f} SDG={epoch_SDG_steps}")
                

    def train(self):
        print("Starting Task Specific Task Residual Adapter training")           
        start_time = time.time()
        
        if self.residual_adapter is None:
            print("No residual_adapter specified, training skipped" )
            return
        
        max_episode_length = int(self.train_config["max_episode_length"])
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        iterations = int(self.train_config["iterations"])
        train_epochs = int(self.train_config["train_epochs"])
        batch_size = int(self.train_config["batch_size"]) 
        data_collection_policy = self.train_config["data_collection_policy"]
    

        for iteration_index in range(iterations):
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            
            self.buffer.reset()
            policy = "random" if iteration_index == 0 else data_collection_policy
            self._collect_env_steps(policy, steps_per_iteration, max_episode_length)
                 
            norm_stats = self.buffer.get_normalization_stats()
            self.residual_adapter.update_normalization_stats(*norm_stats)
            self.residual_adapter.train()
            
            self.train_for_iteration(train_epochs,batch_size)
            self._print_iteration_k_step_rmse()
            


        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"\nTraining finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")  

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
   