from collections import deque
import os
import yaml
from algorithms.base_trainer import BaseTrainer
import numpy as np
import torch
import time
from utils.seed import set_seed
from algorithms.untils import make_dynamics_model
from algorithms.meta_learned_residual_adapter.transition_buffer import TransitionBuffer
from algorithms.meta_learned_residual_adapter.residual_adapter import ResidualAdapter
from algorithms.meta_learned_residual_adapter.residual_dynamics_wrapper import ResidualDynamicsWrapper
from algorithms.meta_learned_residual_adapter.window_sampler import sample_meta_batch
from algorithms.meta_learned_residual_adapter.planner import CrossEntropyMethodPlanner
from algorithms.meta_learned_residual_adapter.model_error import compute_sse_by_dim_for_episode_k
from algorithms.meta_learned_residual_adapter.logging_utils import (
    compute_meta_update_stats,
    format_meta_update_line,
)

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
        return self.make_planner(planner_config, dynamics_fn, reward_fn, action_space, self.device, self.train_seed)

    def make_planner(self, planner_config, dynamics_fn, reward_fn, action_space, device, seed):
        if planner_config is None:
            raise AttributeError("Missing planner config in YAML")
        
        planner_type = planner_config.get("type")         
        horizon = int(planner_config.get("horizon"))
        n_candidates = int(planner_config.get("n_candidates"))
        discount = float(planner_config.get("discount"))
        
        act_low = action_space.low
        act_high = action_space.high
                
        if planner_type == "cem":
            num_cem_iters = int(planner_config.get("num_cem_iters"))
            percent_elites = float(planner_config.get("percent_elites"))
            alpha = float(planner_config.get("alpha"))        
            return CrossEntropyMethodPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount, num_cem_iters, percent_elites, alpha, seed)
        
        raise AttributeError(f"Planner type {planner_type} not supported")
    
    
    
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
        log_episode_forward_progress = []
        
        while steps_collected_this_iteration < steps_target:
            obs, _ = self.env.reset()            
            log_episodes += 1
            episode_return = 0.0
            episode_x_start = None
            episode_x_last = None

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
                
                x_position = float(self._get_forward_position(info))
                if episode_x_start is None:
                    episode_x_start = x_position
                episode_x_last = x_position
                
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
            log_episode_forward_progress.append(float(episode_x_last - episode_x_start))


        self.buffer.add_trajectories(all_transitions)
            
        
        reward_mean = float(np.mean(log_episode_returns))
        reward_std = float(np.std(log_episode_returns))
        
        forward_mean = float(np.mean(log_episode_forward_progress))
        forward_std = float(np.std(log_episode_forward_progress))
                
        log_collect_time = time.time() - log_collect_start_time
        log_episodes = log_episodes
        
        print(f"collect:" 
              f"steps={steps_collected_this_iteration} " 
              f"episodes={log_episodes} " 
              f"reward ={reward_mean:.3f}  ± {reward_std:.3f}  " 
              f"forward_mean={forward_mean:.3f} ± {forward_std:.3f} "
              f"time={log_collect_time:.1f}s"
              )
        
        return {
            "steps": steps_collected_this_iteration,
            "episodes": log_episodes,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "forward_mean": forward_mean,
            "forward_std": forward_std,
            "collect_time_sec": float(log_collect_time),
        }

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
        
        iter_metrics = []

        # Retrive parameters from the yaml file
        max_episode_length = int(self.train_config["max_episode_length"])
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        iterations = int(self.train_config["iterations"])
        meta_updates_per_iter = int(self.train_config["meta_updates_per_iter"])
        meta_batch_size = int(self.train_config["meta_batch_size"])

        for iteration_index in range(iterations):
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            # Collect rollouts using the current wrapper (base+residual)
            collect_stats = self._collect_env_steps(iteration_index, steps_per_iteration, max_episode_length)
            # Run multiple meta-gradient updates using the current buffer data
            meta_update_stats = []
            for meta_update_idx in range(meta_updates_per_iter):
                # Randomly sample a meta-batch of support/query windows from the buffer
                batch = self._sample_batch("train")
                # Move the sampled batch to torch tensors on the correct device
                bt = self._to_torch(batch)

                # Initialize lists to store per-window query losses and support losses
                query_losses = []
                query_losses_pre = []
                support_losses = []

                # Loop over each window in the meta-batch
                for i in range(meta_batch_size):
                    # Inner loop: compute temporary adapted parameters from this support window
                    adapted = self.residual_dynamics_wrapper.compute_adapted_params(bt["support_obs"][i], bt["support_act"][i], bt["support_next_obs"][i], track_higher_grads=True)
                    
                    # Compute support loss using the residual adapter prior parameters
                    support_loss_i = self._compute_loss(bt["support_obs"][i], bt["support_act"][i], bt["support_next_obs"][i], None)
                    # Compute query loss before adaptation (prior parameters)
                    q_loss_pre_i = self._compute_loss(bt["query_obs"][i], bt["query_act"][i], bt["query_next_obs"][i], None)
                    # Compute query loss using the temp adapted parameters
                    q_loss_i = self._compute_loss(bt["query_obs"][i], bt["query_act"][i], bt["query_next_obs"][i], adapted)

                    # Store loss for SGD and logging 
                    support_losses.append(support_loss_i.detach())
                    query_losses_pre.append(q_loss_pre_i.detach())
                    query_losses.append(q_loss_i)

                meta_loss = torch.stack(query_losses).mean()
                self.optimizer.zero_grad()
                # keep the graph for the duration of this meta-update in case
                # any subsequent backward in this loop reuses it
                meta_loss.backward(retain_graph=True)
                self.optimizer.step()
                log_prefix = f"[upd {meta_update_idx}] "
                stats = compute_meta_update_stats(
                    support_losses=support_losses,
                    query_losses_pre=query_losses_pre,
                    query_losses_post=query_losses,
                    residual_adapter=self.residual_adapter,
                )
                meta_update_stats.append(stats)
                line = format_meta_update_line(stats, prefix=log_prefix)
                print(line)
            
            iter_metrics.append(
                self._build_iteration_metrics(iteration_index, collect_stats, meta_update_stats)
            )

                
        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"\nTraining finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")  
        self._print_training_summary_and_recommendations(iter_metrics)

    def evaluate(self):
        print("Overwriting base evaluate to predict model error")
        seeds = self.eval_config["seeds"]
        max_episode_length = int(self.train_config["max_episode_length"])
        k_list = self.eval_config["k_list"]

        
        eval_start_time = time.time()
        episode_rewards = []
        episode_forward_progresses = []

        base_sse_total_by_k = {k: None for k in k_list}
        ra_sse_total_by_k = {k: None for k in k_list}
        base_count_total_by_k = {k: 0 for k in k_list}
        ra_count_total_by_k = {k: 0 for k in k_list}

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

            for k in k_list:
                base_sse, base_count = compute_sse_by_dim_for_episode_k(
                    episode_transitions, self.pretrained_dynamics_model, k, self.device
                )
                if base_sse is not None and base_count > 0:
                    if base_sse_total_by_k[k] is None:
                        base_sse_total_by_k[k] = base_sse.clone()
                    else:
                        base_sse_total_by_k[k] += base_sse
                    base_count_total_by_k[k] += base_count

                if self.residual_adapter is not None:
                    ra_sse, ra_count = compute_sse_by_dim_for_episode_k(
                        episode_transitions, self.residual_dynamics_wrapper, k, self.device
                    )
                    if ra_sse is not None and ra_count > 0:
                        if ra_sse_total_by_k[k] is None:
                            ra_sse_total_by_k[k] = ra_sse.clone()
                        else:
                            ra_sse_total_by_k[k] += ra_sse
                        ra_count_total_by_k[k] += ra_count

            env.close()
            
            print(f"\n------------------[seed {seed}]------------------")
            print(f"[rollout] reward={ep_reward:.4f} forward_progress={forward_progress:.4f} len={steps}")
            
     
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

        for k in k_list:
            if base_sse_total_by_k[k] is not None and base_count_total_by_k[k] > 0:
                base_rmse_summary = torch.sqrt(base_sse_total_by_k[k] / base_count_total_by_k[k]).tolist()
            else:
                base_rmse_summary = None

            if ra_sse_total_by_k[k] is not None and ra_count_total_by_k[k] > 0:
                ra_rmse_summary = torch.sqrt(ra_sse_total_by_k[k] / ra_count_total_by_k[k]).tolist()
            else:
                ra_rmse_summary = None

            self._print_rmse_table(
                f"RMSE mean by dim (k-{k}) [all episodes]",
                base_rmse_summary,
                ra_rmse_summary,
            )
            print()
            
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

    def _print_rmse_table(self, title, base_rmse, ra_rmse, label_width=10, num_width=7):
        print(title)
        if base_rmse is None or ra_rmse is None:
            print(f"{'BASE':<{label_width}} n/a")
            print(f"{'BASE+RA':<{label_width}} n/a")
            return

        base_list = [float(v) for v in base_rmse]
        ra_list = [float(v) for v in ra_rmse]
        num_dims = min(len(base_list), len(ra_list))
        if num_dims == 0:
            print(f"{'BASE':<{label_width}} n/a")
            print(f"{'BASE+RA':<{label_width}} n/a")
            return

        dims = range(0, num_dims)
        header = f"{'DIM.':<{label_width}}" + "".join([f"{d:>{num_width}d}" for d in dims])
        base_line = f"{'BASE':<{label_width}}" + "".join([f"{base_list[d]:>{num_width}.3f}" for d in dims])
        ra_line = f"{'BASE+RA':<{label_width}}" + "".join([f"{ra_list[d]:>{num_width}.3f}" for d in dims])
        print(header)
        print(base_line)
        print(ra_line)

    def _aggregate_meta_update_stats(self, stats_list):
        def _mean(key):
            vals = [s[key] for s in stats_list if s.get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")

        def _std(key):
            vals = [s[key] for s in stats_list if s.get(key) is not None]
            return float(np.std(vals)) if vals else float("nan")

        return {
            "support_mean": _mean("support_mean"),
            "support_std": _mean("support_std"),
            "query_pre_mean": _mean("query_pre_mean"),
            "query_post_mean": _mean("query_post_mean"),
            "query_post_std_mean": _mean("query_post_std"),
            "query_post_std": _std("query_post_mean"),
            "query_improve_mean": _mean("query_improve"),
            "param_norm_mean": _mean("param_norm"),
            "grad_norm_mean": _mean("grad_norm"),
        }

    def _build_iteration_metrics(self, iteration_index, collect_stats, meta_update_stats):
        meta_stats = self._aggregate_meta_update_stats(meta_update_stats)
        return {
            "iteration": iteration_index,
            **collect_stats,
            **meta_stats,
        }

    def _print_training_summary_and_recommendations(self, iter_metrics):
        if len(iter_metrics) == 0:
            print("\n[summary] no iteration metrics collected")
            return

        reward_series = [m["reward_mean"] for m in iter_metrics]
        error_series = [m["query_post_mean"] for m in iter_metrics]
        improve_series = [m["query_improve_mean"] for m in iter_metrics]

        reward_finite = [v for v in reward_series if np.isfinite(v)]
        error_finite = [v for v in error_series if np.isfinite(v)]
        improve_finite = [v for v in improve_series if np.isfinite(v)]

        reward_last = reward_series[-1]
        reward_best = max(reward_finite) if reward_finite else float("nan")
        error_last = error_series[-1]
        error_best = min(error_finite) if error_finite else float("nan")
        improve_last = improve_series[-1] if improve_series else float("nan")
        if improve_finite:
            improve_last = improve_finite[-1]

        print("\n[training summary]")
        print(f"- reward_mean: last={reward_last:.3f} best={reward_best:.3f}")
        print(f"- query_post_mean: last={error_last:.6f} best={error_best:.6f}")
        print(f"- query_improve_mean: last={improve_last:.6f}")

        self._print_hparam_suggestions(iter_metrics)

    def _print_hparam_suggestions(self, iter_metrics):
        def _safe_div(a, b, default=0.0):
            if b == 0.0 or not np.isfinite(b):
                return default
            return a / b
        
        def _finite_or(value, default=0.0):
            return value if np.isfinite(value) else default

        def _cv(values):
            if len(values) < 2:
                return 0.0
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            return std_val / mean_val if mean_val != 0 else 0.0

        reward_series = [m["reward_mean"] for m in iter_metrics if np.isfinite(m["reward_mean"])]
        error_series = [m["query_post_mean"] for m in iter_metrics if np.isfinite(m["query_post_mean"])]
        if len(reward_series) == 0 or len(error_series) == 0:
            print("\n[hyperparam suggestions]")
            print("- insufficient metrics to compute recommendations")
            return
        reward_last = reward_series[-1]
        reward_best = max(reward_series)
        error_last = error_series[-1]
        error_best = min(error_series)

        reward_improving = reward_last >= reward_best * 0.98
        error_improving = error_last <= error_best * 1.02

        last = iter_metrics[-1]
        query_pre = _finite_or(last.get("query_pre_mean", float("nan")))
        query_post = _finite_or(last.get("query_post_mean", float("nan")))
        improvement = _finite_or(last.get("query_improve_mean", float("nan")))
        support_mean = _finite_or(last.get("support_mean", float("nan")))

        improvement_ratio = _safe_div(improvement, max(query_pre, 1e-8))
        gap_ratio = _safe_div(support_mean, max(query_post, 1e-8))

        reward_cv = _cv([m["reward_mean"] for m in iter_metrics[-5:]])
        error_cv = _cv([m["query_post_mean"] for m in iter_metrics[-5:]])

        # iterations
        if reward_improving or error_improving:
            iter_rec = "increase"
        else:
            iter_rec = "decrease"

        # steps_per_iteration
        if reward_cv > 0.5 or (error_improving and not reward_improving):
            steps_rec = "increase"
        else:
            steps_rec = "decrease"

        # hidden_sizes
        if gap_ratio < 0.7:
            hidden_rec = "decrease"
        else:
            hidden_rec = "increase"

        # support_length / query_length
        if improvement_ratio < 0.02:
            support_rec = "increase"
            query_rec = "increase"
        else:
            support_rec = "decrease"
            query_rec = "decrease"

        # meta_batch_size
        if last.get("query_post_std", 0.0) > 0.1 * max(query_post, 1e-8):
            meta_batch_rec = "increase"
        else:
            meta_batch_rec = "decrease"

        # inner_steps
        if improvement_ratio < 0.0:
            inner_steps_rec = "decrease"
        elif improvement_ratio < 0.02:
            inner_steps_rec = "increase"
        else:
            inner_steps_rec = "decrease"

        # inner_learning_rate
        if improvement_ratio < 0.0:
            inner_lr_rec = "decrease"
        elif improvement_ratio < 0.02:
            inner_lr_rec = "increase"
        else:
            inner_lr_rec = "decrease"

        # outer_learning_rate
        if error_cv > 0.1 or not error_improving:
            outer_lr_rec = "decrease"
        else:
            outer_lr_rec = "increase"

        # meta_updates_per_iter
        if not error_improving and improvement_ratio > 0.0:
            meta_updates_rec = "increase"
        else:
            meta_updates_rec = "decrease"

        print("\n[hyperparam suggestions]")
        print(f"- iterations: {iter_rec}")
        print(f"- steps_per_iteration: {steps_rec}")
        print(f"- hidden_sizes: {hidden_rec}")
        print(f"- support_length: {support_rec}")
        print(f"- query_length: {query_rec}")
        print(f"- meta_batch_size: {meta_batch_rec}")
        print(f"- inner_steps: {inner_steps_rec}")
        print(f"- inner_learning_rate: {inner_lr_rec}")
        print(f"- outer_learning_rate: {outer_lr_rec}")
        print(f"- meta_updates_per_iter: {meta_updates_rec}")
       
    def save(self):
        if self.residual_adapter is None:
            print("Residual adapter is not initialized.")
            return
            
        save_path = os.path.join(self.output_dir, "residual_adapter.pt")

        payload = {
            "residual_adapter_state": self.residual_adapter.state_dict(),
        }
        if self.optimizer is not None:
            payload["optimizer_state"] = self.optimizer.state_dict()

        torch.save(payload, save_path)
        print(f"Residual adapter saved to {save_path}")

    def load(self, path):
        if self.residual_adapter is None:
            raise RuntimeError("Residual adapter is not initialized.")

        model_path = path
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "residual_adapter.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get("residual_adapter_state", checkpoint)
        self.residual_adapter.load_state_dict(state_dict)

        if self.optimizer is not None and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        print(f"Loaded residual adapter from {model_path}")
        return self
   
