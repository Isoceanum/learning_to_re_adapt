from collections import deque
import yaml
from algorithms.base_trainer import BaseTrainer
import numpy as np
import torch
import time
from utils.seed import set_seed
from algorithms.untils import make_dynamics_model, make_planner
from evaluation.model_error import compute_k_step_rmse_for_episode, compute_top_rmse_by_dim_for_episode
from algorithms.meta_learned_residual_adapter.transition_buffer import TransitionBuffer
from algorithms.meta_learned_residual_adapter.residual_adapter import ResidualAdapter
from algorithms.meta_learned_residual_adapter.residual_dynamics_wrapper import ResidualDynamicsWrapper
from algorithms.meta_learned_residual_adapter.window_sampler import sample_meta_batch

class MetaLearnedResidualAdapterTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        self.env = self._make_train_env()
        
        self.pretrained_dynamics_model = self.load_pretrained_dynamics_model()
        self.buffer = self._make_buffer()
        self.residual_adapter = self._make_residual_adapter()
        self.residual_dynamics_wrapper = self._make_residual_dynamics_wrapper()
        self.planner = self._make_planner()
        self.optimizer = self._make_optimizer()
        
        # Eval-time online adaptation state (used by predict)
        self.eval_adapt_window = None
        self.eval_last_obs = None
        self.eval_last_action = None
          
    def _make_optimizer(self):
        if self.residual_adapter is None:
            return
        learning_rate = float(self.train_config["outer_learning_rate"])
        return torch.optim.Adam(self.residual_adapter.parameters(), lr=learning_rate)
        
    def _make_residual_dynamics_wrapper(self):
        inner_learning_rate = float(self.train_config["inner_learning_rate"])
        inner_steps = int(self.train_config["inner_steps"])
        
        return ResidualDynamicsWrapper(self.pretrained_dynamics_model, inner_steps, inner_learning_rate, self.residual_adapter) 
        
    def _make_residual_adapter(self):   
        residual_adapter_config = self.train_config.get("residual_adapter")
        if residual_adapter_config is None or not residual_adapter_config.get("enabled", False):
            return None
        
        hidden_sizes = residual_adapter_config.get("hidden_sizes")
        return ResidualAdapter(self.env.observation_space.shape[0], self.env.action_space.shape[0], hidden_sizes).to(self.device)
        
    def _make_buffer(self):
        valid_split_ratio = float(self.train_config["valid_split_ratio"])
        return TransitionBuffer(valid_split_ratio, self.train_seed)
     
    def _make_planner(self):
        planner_config = self.train_config.get("planner")
        
        residual_adapter_config = self.train_config.get("residual_adapter")
        dynamics_fn = self.pretrained_dynamics_model.predict_next_state if residual_adapter_config is None else self.residual_dynamics_wrapper.predict_next_state_with_parameters    
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
        support_length = int(self.train_config["support_length"])
        log_collect_start_time = time.time()
        steps_collected_this_iteration = 0
        all_transitions = []

        log_episodes = 0

        log_episode_returns = []
        
        while steps_collected_this_iteration < steps_target:
            obs, _ = self.env.reset()            
            log_episodes += 1
            episode_return = 0.0

            episode_steps = 0
            episode_observations = []
            episode_actions = []
            episode_next_observations = []
            adapt_window = deque(maxlen=support_length)
            
            while episode_steps < max_episode_length and steps_collected_this_iteration < steps_target:
                params = None
                
                if len(adapt_window) == support_length:
                    window_obs, window_act, window_next_obs = zip(*adapt_window)
                    support_obs = np.stack(window_obs, axis=0)
                    support_act = np.stack(window_act, axis=0)
                    support_next_obs = np.stack(window_next_obs, axis=0)
                    
                    with torch.enable_grad():
                        params = self.residual_dynamics_wrapper.compute_adapted_params(support_obs, support_act, support_next_obs, track_higher_grads=False)
                
                action = self.planner.plan(obs, parameters=params)
                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy()
                    
                next_obs, reward, terminated, truncated, info = self._step_env(action)
                episode_return += float(reward)
                
                episode_observations.append(obs)
                episode_actions.append(action)
                episode_next_observations.append(next_obs)
                adapt_window.append((obs, action, next_obs))
                obs = next_obs
            
                episode_steps += 1
                steps_collected_this_iteration += 1
                            
                if terminated or truncated:
                    break
                
            all_transitions.append((
                np.asarray(episode_observations, dtype=np.float32),
                np.asarray(episode_actions, dtype=np.float32),
                np.asarray(episode_next_observations, dtype=np.float32),
            ))
            log_episode_returns.append(float(episode_return))


        self.buffer.add_trajectories(all_transitions)
            
        
        reward_mean = float(np.mean(log_episode_returns))
        reward_std = float(np.std(log_episode_returns))
                
        log_collect_time = time.time() - log_collect_start_time
        log_episodes = log_episodes
        
        print()
        
        print(f"collect:" 
              f"steps={steps_collected_this_iteration} " 
              f"episodes={log_episodes} " 
              f"reward ={reward_mean:.3f}  ± {reward_std:.3f}  " 
              f"time={log_collect_time:.1f}s"
              )

    def _sample_batch(self, split="train"):
        meta_batch_size = int(self.train_config["meta_batch_size"])
        support_length = int(self.train_config["support_length"])
        query_length = int(self.train_config["query_length"])
        return sample_meta_batch(self.buffer, meta_batch_size, support_length, query_length, split)

    def _to_torch(self, batch):
        batch_torch = {}
        for key, value in batch.items():
            tensor = torch.as_tensor(value, dtype=torch.float32)
            tensor = tensor.to(self.device)
            batch_torch[key] = tensor
        return batch_torch
    
    def _normalize_batch(self, batch_torch):
        base = self.pretrained_dynamics_model
        base._assert_normalization_stats()
        batch_norm = {}
        
        batch_norm["support_obs_norm"] = (batch_torch["support_obs"] - base.mean_obs) / base.std_obs
        batch_norm["support_act_norm"] = (batch_torch["support_act"] - base.mean_act) / base.std_act
        batch_norm["support_next_obs_norm"] = (batch_torch["support_next_obs"] - base.mean_obs) / base.std_obs
        
        batch_norm["query_obs_norm"] = (batch_torch["query_obs"] - base.mean_obs) / base.std_obs
        batch_norm["query_act_norm"] = (batch_torch["query_act"] - base.mean_act) / base.std_act
        batch_norm["query_next_obs_norm"] = (batch_torch["query_next_obs"] - base.mean_obs) / base.std_obs

        return batch_norm
        
    def _mse(self, prediction, target):
        return torch.mean((prediction - target) ** 2)

    def _compute_loss(self, obs, act, next_obs, params):
        base = self.pretrained_dynamics_model
        base._assert_normalization_stats()

        # Predict next state using BASE + (residual adapter with optional adapted params)
        pred_next = self.residual_dynamics_wrapper.predict_next_state_with_parameters(obs, act, parameters=params)  # raw

        # Compare in normalized next-state space (consistent with residual adapter design)
        pred_next_norm = (pred_next - base.mean_obs) / base.std_obs
        next_obs_norm = (next_obs - base.mean_obs) / base.std_obs

        return self._mse(pred_next_norm, next_obs_norm)
        
    def _flat(self, x):
        return x.reshape(-1, x.shape[-1])

    def train(self):
        print("Starting Meta Learned Residual Adapter  training")           
        start_time = time.time()
        
        if self.residual_adapter is None:
            print("No residual adpater defined in yaml, skipping training")
            return

        # Retrive parameters from the yaml file
        max_episode_length = int(self.train_config["max_episode_length"])
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        iterations = int(self.train_config["iterations"])
        meta_updates_per_iter = int(self.train_config["meta_updates_per_iter"])
        meta_batch_size = int(self.train_config["meta_batch_size"])

        for iteration_index in range(iterations):
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            # Collect rollouts using the current wrapper (base+residual)
            self._collect_env_steps(iteration_index, steps_per_iteration, max_episode_length)
            # Run multiple meta-gradient updates using the current buffer data
            for meta_update_idx in range(meta_updates_per_iter):
                # Randomly sample a meta-batch of support/query windows from the buffer
                batch = self._sample_batch("train")
                # Move the sampled batch to torch tensors on the correct device
                bt = self._to_torch(batch)

                # Initialize lists to store per-window query losses and support losses
                query_losses = []
                support_losses = []

                # Loop over each window in the meta-batch
                for i in range(meta_batch_size):
                    # Inner loop: compute temporary adapted parameters from this support window
                    adapted = self.residual_dynamics_wrapper.compute_adapted_params(bt["support_obs"][i], bt["support_act"][i], bt["support_next_obs"][i], track_higher_grads=True)
                    
                    # Compute support loss using the residual adapter prior parameters
                    support_loss_i = self._compute_loss(bt["support_obs"][i], bt["support_act"][i], bt["support_next_obs"][i], None)
                    # Compute query loss using the temp adapted parameters
                    q_loss_i = self._compute_loss(bt["query_obs"][i], bt["query_act"][i], bt["query_next_obs"][i], adapted)

                    # Store loss for SGD and logging 
                    support_losses.append(support_loss_i.detach())
                    query_losses.append(q_loss_i)

                meta_loss = torch.stack(query_losses).mean()
                self.optimizer.zero_grad()
                # keep the graph for the duration of this meta-update in case
                # any subsequent backward in this loop reuses it
                meta_loss.backward(retain_graph=True)
                self.optimizer.step()

                
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
            # Reset any eval-time adaptation state so seeds don't leak information.
            self._reset_eval_adaptation()
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
            base_rmse_by_k = compute_k_step_rmse_for_episode(
                episode_transitions,
                self.pretrained_dynamics_model,
                k_list,
                self.device,
                adapt=False,
            )
            for k in k_list:
                base_rmse_values_by_k[k].append(base_rmse_by_k[k])

            print("[BASE]    RMSE:", " | ".join([f"k-{k} {base_rmse_by_k[k]:.4f}" for k in k_list]))
            
            ra_rmse_by_k = compute_k_step_rmse_for_episode(
                episode_transitions,
                self.residual_dynamics_wrapper,
                k_list,
                self.device,
                adapt=True,
                support_length=int(self.train_config["support_length"]),
            )
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
        
        if self.residual_adapter is None:
            action = self.planner.plan(obs_t, parameters=None)
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            return action
            
        support_length = int(self.train_config["support_length"])
        if self.eval_adapt_window is None:
            self.eval_adapt_window = deque(maxlen=support_length)
            self.eval_last_obs = None
            self.eval_last_action = None
            
        # add the last transition into the window (so we can adapt on recent real data)
        if self.eval_last_obs is not None and self.eval_last_action is not None:
            self.eval_adapt_window.append((self.eval_last_obs, self.eval_last_action, obs))
            
        params = None
        if len(self.eval_adapt_window) == support_length:
            window_obs, window_act, window_next_obs = zip(*self.eval_adapt_window)
            support_obs = np.stack(window_obs, axis=0)
            support_act = np.stack(window_act, axis=0)
            support_next_obs = np.stack(window_next_obs, axis=0)
            with torch.enable_grad():
                params = self.residual_dynamics_wrapper.compute_adapted_params(support_obs, support_act, support_next_obs, track_higher_grads=False)

        
        action = self.planner.plan(obs_t, parameters=params)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
            
        self.eval_last_obs = obs
        self.eval_last_action = action
        return action
    
    
    def _reset_eval_adaptation(self):
        self.eval_adapt_window = None
        self.eval_last_obs = None
        self.eval_last_action = None
       
    def save(self):
        print("no saving for now")
   
