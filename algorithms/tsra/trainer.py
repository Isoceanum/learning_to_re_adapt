
import os
import yaml
from algorithms.base_trainer import BaseTrainer
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
from typing import Dict, List, Tuple
from utils.seed import set_seed
from algorithms.untils import make_dynamics_model
from evaluation.model_error import (
    compute_k_step_rmse_for_episode,
    compute_sse_by_dim_for_episode_k,
)
from algorithms.tsra.residual_adapter import ResidualAdapter
from algorithms.tsra.planner import CrossEntropyMethodPlanner, RandomShootingPlanner, MPPIPlanner
from algorithms.tsra.residual_dynamics_wrapper import ResidualDynamicsWrapper
from torch.utils.data import DataLoader, TensorDataset


def _summary_stats(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    mean_signed = float(np.mean(arr))
    mae = float(np.mean(np.abs(arr)))
    std = float(np.std(arr))
    return mean_signed, mae, std


def _k_step_dim_error_stats_for_episode(
    episode_transitions,
    model,
    k_targets: List[int],
    device,
    dim_idx: int,
):
    if not k_targets:
        return {}, {}

    k_max = max(k_targets)
    num_transitions = len(episode_transitions)
    errors_by_k: Dict[int, List[float]] = {k: [] for k in k_targets}
    bin_labels = [
        "vx<0",
        "0<=vx<0.5",
        "0.5<=vx<1.0",
        "vx>=1.0",
    ]
    binned_errors: Dict[int, Dict[str, List[float]]] = {
        k: {label: [] for label in bin_labels} for k in k_targets
    }

    if num_transitions < k_max:
        return errors_by_k, binned_errors

    for start in range(0, num_transitions - k_max + 1):
        start_obs = episode_transitions[start][0]
        pred_obs = torch.as_tensor(start_obs, dtype=torch.float32, device=device).unsqueeze(0)

        for step in range(1, k_max + 1):
            action = episode_transitions[start + step - 1][1]
            true_next_obs = episode_transitions[start + step - 1][2]

            action_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            true_next_obs_t = torch.as_tensor(true_next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            pred_next_obs = model.predict_next_state(pred_obs, action_t)

            if step in errors_by_k:
                pred_vx = float(pred_next_obs[0, dim_idx].detach().cpu().item())
                true_vx = float(true_next_obs_t[0, dim_idx].detach().cpu().item())
                signed_err = pred_vx - true_vx
                errors_by_k[step].append(signed_err)

                if true_vx < 0.0:
                    label = "vx<0"
                elif true_vx < 0.5:
                    label = "0<=vx<0.5"
                elif true_vx < 1.0:
                    label = "0.5<=vx<1.0"
                else:
                    label = "vx>=1.0"
                binned_errors[step][label].append(signed_err)

            pred_obs = pred_next_obs

    return errors_by_k, binned_errors


class TSRATrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        self.env = self._make_train_env() # make the env used for training
        self.pretrained_dynamics_model = self.load_pretrained_dynamics_model() # load a pretrained dynamics model
        self.residual_adapter = self._make_residual_adapter() # make residual adapter
        self.optimizer = self._make_optimizer()
        self.residual_dynamics_wrapper = self._make_residual_dynamics_wrapper() # make base + residual adapter wrapper 
        self.planner = self._make_planner() # make planner 
        self.base_planner = self._make_base_planner() # base-only planner for bootstrap
        
    def _make_residual_adapter(self):   
        residual_adapter_config = self.train_config.get("residual_adapter")
        if residual_adapter_config is None or not residual_adapter_config.get("enabled", False):
            return None
        
        hidden_sizes = residual_adapter_config.get("hidden_sizes")
        return ResidualAdapter(self.env.observation_space.shape[0], self.env.action_space.shape[0], hidden_sizes).to(self.device)

    def _make_optimizer(self):
        if self.residual_adapter is None:
            return None
        residual_adapter_config = self.train_config.get("residual_adapter", {})
        learning_rate = float(residual_adapter_config.get("learning_rate", 1e-3))
        return optim.AdamW(self.residual_adapter.parameters(), lr=learning_rate)
        
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
        
        
        if planner_type == "rs":
            return RandomShootingPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, self.device, discount)
            
            
        if planner_type == "mppi": 
            return MPPIPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, self.device)
            
        
        raise AttributeError(f"Planner type {planner_type} not supported")

    def _make_residual_dynamics_wrapper(self):
        return ResidualDynamicsWrapper(self.pretrained_dynamics_model, self.residual_adapter) 
        
    def _make_planner(self):
        planner_config = self.train_config.get("planner")
        
        dynamics_fn = self.pretrained_dynamics_model.predict_next_state if self.residual_adapter is None else self.residual_dynamics_wrapper.predict_next_state    
        action_space = self.env.action_space
        
        base_env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(base_env, "get_model_reward_fn"): 
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")
        
        reward_fn = base_env.get_model_reward_fn()
        return self.make_planner(planner_config, dynamics_fn, reward_fn, action_space, self.device, self.train_seed)

    def _make_base_planner(self):
        planner_config = self.train_config.get("planner")
        action_space = self.env.action_space

        base_env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(base_env, "get_model_reward_fn"):
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")

        reward_fn = base_env.get_model_reward_fn()
        return self.make_planner(planner_config, self.pretrained_dynamics_model.predict_next_state, reward_fn, action_space, self.device, self.train_seed)
    
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
    
    def _collect_env_steps(self, steps_target, max_episode_length, use_base_only=False):
        collect_start_time = time.time()
        
        steps_collected_this_iteration = 0
        log_episodes = 0
        log_episode_returns = []
        log_episode_forward_progress = []
        
        obs_all = []
        act_all = []
        next_obs_all = []
        
        while steps_collected_this_iteration < steps_target:
            obs, _ = self.env.reset()
            log_episodes += 1
            
            episode_return = 0.0
            episode_x_start = None
            episode_x_last = None
            episode_steps = 0            
            
            while episode_steps < max_episode_length:
                planner = self.base_planner if (use_base_only and self.base_planner is not None) else self.planner
                action = planner.plan(obs)
                    
                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy()
                    
                next_obs, reward, terminated, truncated, info = self._step_env(action)  
                episode_return += float(reward)
                
                x_position = float(self._get_forward_position(info))
                if episode_x_start is None:
                    episode_x_start = x_position
                episode_x_last = x_position
                
                obs_all.append(obs)
                act_all.append(action)
                next_obs_all.append(next_obs)
                obs = next_obs
            
                episode_steps += 1
                steps_collected_this_iteration += 1
                            
                if steps_collected_this_iteration >= steps_target or terminated or truncated:
                    break
                
            log_episode_forward_progress.append(float(episode_x_last - episode_x_start))
            log_episode_returns.append(float(episode_return))
                
        log_collect_time = time.time() - collect_start_time
        reward_mean = float(np.mean(log_episode_returns))
        reward_std = float(np.std(log_episode_returns))
        
        forward_mean = float(np.mean(log_episode_forward_progress))
        forward_std = float(np.std(log_episode_forward_progress))
                
        print(f"Collected: " 
              f"steps={steps_collected_this_iteration} "
              f"reward_mean={reward_mean:.3f} ± {reward_std:.3f} " 
              f"forward_mean={forward_mean:.3f} ± {forward_std:.3f} "
              f"time={log_collect_time:.1f}s")
        
        return obs_all, act_all, next_obs_all

    def _run_epoch(self, loader, train):
        if self.residual_adapter is None:
            raise RuntimeError("Residual adapter is not initialized.")
        if self.optimizer is None and train:
            raise RuntimeError("Optimizer is not initialized.")

        loss_fn = nn.SmoothL1Loss(reduction="mean")
        base = self.pretrained_dynamics_model
        base_mean_obs = base.mean_obs
        base_std_obs = base.std_obs
        base_mean_act = base.mean_act
        base_std_act = base.std_act
        eps = 1e-8

        stats = {
            "loss": [],
            "base_mse": [],
            "pred_mse": [],
            "corr_norm": [],
            "corr_ratio": [],
        }

        self.residual_adapter.train() if train else self.residual_adapter.eval()

        def _accumulate(loss, base_mse, pred_mse, corr_norm, corr_ratio):
            stats["loss"].append(float(loss))
            stats["base_mse"].append(float(base_mse))
            stats["pred_mse"].append(float(pred_mse))
            stats["corr_norm"].append(float(corr_norm))
            stats["corr_ratio"].append(float(corr_ratio))

        if train:
            for obs_b, act_b, next_obs_b in loader:
                self.optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    base_pred_next = base.predict_next_state(obs_b, act_b)

                obs_norm = (obs_b - base_mean_obs) / base_std_obs
                act_norm = (act_b - base_mean_act) / base_std_act
                base_pred_next_norm = (base_pred_next - base_mean_obs) / base_std_obs
                next_obs_norm = (next_obs_b - base_mean_obs) / base_std_obs

                correction_norm = self.residual_adapter(obs_norm, act_norm, base_pred_next_norm)
                pred_next_norm = base_pred_next_norm + correction_norm

                loss = loss_fn(pred_next_norm, next_obs_norm)
                loss.backward()
                self.optimizer.step()

                base_mse = torch.mean((base_pred_next_norm - next_obs_norm) ** 2)
                pred_mse = torch.mean((pred_next_norm - next_obs_norm) ** 2)
                corr_norm = torch.norm(correction_norm, dim=1).mean()
                base_norm = torch.norm(base_pred_next_norm, dim=1)
                corr_ratio = (torch.norm(correction_norm, dim=1) / (base_norm + eps)).mean()

                _accumulate(loss.item(), base_mse.item(), pred_mse.item(), corr_norm.item(), corr_ratio.item())
        else:
            with torch.no_grad():
                for obs_b, act_b, next_obs_b in loader:
                    base_pred_next = base.predict_next_state(obs_b, act_b)

                    obs_norm = (obs_b - base_mean_obs) / base_std_obs
                    act_norm = (act_b - base_mean_act) / base_std_act
                    base_pred_next_norm = (base_pred_next - base_mean_obs) / base_std_obs
                    next_obs_norm = (next_obs_b - base_mean_obs) / base_std_obs

                    correction_norm = self.residual_adapter(obs_norm, act_norm, base_pred_next_norm)
                    pred_next_norm = base_pred_next_norm + correction_norm

                    loss = loss_fn(pred_next_norm, next_obs_norm)
                    base_mse = torch.mean((base_pred_next_norm - next_obs_norm) ** 2)
                    pred_mse = torch.mean((pred_next_norm - next_obs_norm) ** 2)
                    corr_norm = torch.norm(correction_norm, dim=1).mean()
                    base_norm = torch.norm(base_pred_next_norm, dim=1)
                    corr_ratio = (torch.norm(correction_norm, dim=1) / (base_norm + eps)).mean()

                    _accumulate(loss.item(), base_mse.item(), pred_mse.item(), corr_norm.item(), corr_ratio.item())

        if len(stats["loss"]) == 0:
            return {k: float("nan") for k in stats}

        return {k: float(np.mean(v)) for k, v in stats.items()}

    def _quick_eval_rollout(self, episodes, max_episode_length):
        if episodes <= 0:
            return

        eval_seed = int(self.eval_config.get("seeds", [self.train_seed])[0])
        set_seed(eval_seed)
        env = self._make_eval_env(seed=eval_seed)

        episode_rewards = []
        episode_forward = []
        for _ in range(episodes):
            obs, _ = env.reset(seed=eval_seed)
            done = False
            steps = 0
            ep_reward = 0.0
            x_start = None
            x_last = None

            while not done and steps < max_episode_length:
                action = self.predict(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += float(reward)
                steps += 1

                x_position = float(self._get_forward_position(info))
                if x_start is None:
                    x_start = x_position
                x_last = x_position
                obs = next_obs

            episode_rewards.append(ep_reward)
            episode_forward.append(float(x_last - x_start) if (x_start is not None and x_last is not None) else 0.0)

        env.close()
        reward_mean = float(np.mean(episode_rewards))
        reward_std = float(np.std(episode_rewards))
        forward_mean = float(np.mean(episode_forward))
        forward_std = float(np.std(episode_forward))
        print(f"[quick-eval] episodes={episodes} reward_mean={reward_mean:.3f} ± {reward_std:.3f} forward_mean={forward_mean:.3f} ± {forward_std:.3f}")

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

    def _eval_rmse_by_dim_tables(self, seeds, k_dim_list, max_episode_length, header_prefix, per_seed=True):
        if not k_dim_list:
            return

        base_sse_total_by_k = {k: None for k in k_dim_list}
        ra_sse_total_by_k = {k: None for k in k_dim_list}
        base_count_total_by_k = {k: 0 for k in k_dim_list}
        ra_count_total_by_k = {k: 0 for k in k_dim_list}

        for seed in seeds:
            set_seed(seed)
            env = self._make_eval_env(seed=seed)

            obs, _ = env.reset(seed=seed)
            done = False
            steps = 0
            episode_transitions = []

            while not done and steps < max_episode_length:
                action = self.predict(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_transitions.append((obs, action, next_obs))
                obs = next_obs
                done = terminated or truncated
                steps += 1

            env.close()

            for k in k_dim_list:
                base_sse, base_count = compute_sse_by_dim_for_episode_k(
                    episode_transitions, self.pretrained_dynamics_model, k, self.device
                )
                ra_sse, ra_count = compute_sse_by_dim_for_episode_k(
                    episode_transitions, self.residual_dynamics_wrapper, k, self.device
                )

                if base_sse is not None and base_count > 0:
                    base_rmse_by_dim = torch.sqrt(base_sse / base_count).tolist()
                    if base_sse_total_by_k[k] is None:
                        base_sse_total_by_k[k] = base_sse.clone()
                    else:
                        base_sse_total_by_k[k] += base_sse
                    base_count_total_by_k[k] += base_count
                else:
                    base_rmse_by_dim = None

                if ra_sse is not None and ra_count > 0:
                    ra_rmse_by_dim = torch.sqrt(ra_sse / ra_count).tolist()
                    if ra_sse_total_by_k[k] is None:
                        ra_sse_total_by_k[k] = ra_sse.clone()
                    else:
                        ra_sse_total_by_k[k] += ra_sse
                    ra_count_total_by_k[k] += ra_count
                else:
                    ra_rmse_by_dim = None

                if per_seed:
                    self._print_rmse_table(
                        f"RMSE mean by dim (k-{k}) [{header_prefix} seed {seed}]",
                        base_rmse_by_dim,
                        ra_rmse_by_dim,
                    )
                    print()

        for k in k_dim_list:
            if base_sse_total_by_k[k] is not None and base_count_total_by_k[k] > 0:
                base_rmse_summary = torch.sqrt(base_sse_total_by_k[k] / base_count_total_by_k[k]).tolist()
            else:
                base_rmse_summary = None

            if ra_sse_total_by_k[k] is not None and ra_count_total_by_k[k] > 0:
                ra_rmse_summary = torch.sqrt(ra_sse_total_by_k[k] / ra_count_total_by_k[k]).tolist()
            else:
                ra_rmse_summary = None

            self._print_rmse_table(
                f"RMSE mean by dim (k-{k}) [{header_prefix} all episodes]",
                base_rmse_summary,
                ra_rmse_summary,
            )
            print()
                
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

    def train(self):
        print("Starting Task Specific Task Residual Adapter training")           
        start_time = time.time()

        print_rmse_each_iteration = False
        
        if self.residual_adapter is None:
            print("No residual_adapter specified, training skipped" )
            return
        
        max_episode_length = int(self.train_config["max_episode_length"])
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        iterations = int(self.train_config["iterations"])
        train_epochs = int(self.train_config["train_epochs"])
        bootstrap_base_only = bool(self.train_config.get("bootstrap_base_only", True))
        quick_eval_episodes = int(self.train_config.get("quick_eval_episodes", 0))
        eval_seeds = self.eval_config.get("seeds", [self.train_seed])
        eval_k_list = self.eval_config.get("k_list", [1, 5, 10, 15])
        max_k = max(eval_k_list) if len(eval_k_list) > 0 else 1
        k_dim_list = [k for k in [1, 5, 10, 15] if k <= max_k]
        if len(k_dim_list) == 0:
            k_dim_list = [1]
        
        iter_val_last = []
        iter_train_last = []
        iter_val_best = []
        last_iter_val_series = []
        last_iter_train_series = []

        for iteration_index in range(iterations):
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            use_base_only = bootstrap_base_only and iteration_index == 0
            if use_base_only:
                print("Collecting rollout data with base-only planner (bootstrap)")
            else:
                print("Collecting rollout data with base+adapter planner")

            obs_all, act_all, next_obs_all = self._collect_env_steps(
                steps_per_iteration,
                max_episode_length,
                use_base_only=use_base_only,
            )
            train_loader, val_loader = self._split_data(obs_all, act_all, next_obs_all)
            
            val_series = []
            train_series = []
            for epoch in range(train_epochs):
                train_stats = self._run_epoch(train_loader, train=True)
                val_stats = self._run_epoch(val_loader, train=False)
                train_series.append(float(train_stats["pred_mse"]))
                val_series.append(float(val_stats["pred_mse"]))
                print(
                    f"epoch {epoch+1}/{train_epochs} "
                    f"train_base_mse={train_stats['base_mse']:.6f} "
                    f"train_pred_mse={train_stats['pred_mse']:.6f} "
                    f"train_corr_norm={train_stats['corr_norm']:.6f} "
                    f"train_corr_ratio={train_stats['corr_ratio']:.6f} "
                    f"val_base_mse={val_stats['base_mse']:.6f} "
                    f"val_pred_mse={val_stats['pred_mse']:.6f} "
                    f"val_corr_norm={val_stats['corr_norm']:.6f} "
                    f"val_corr_ratio={val_stats['corr_ratio']:.6f}"
                )

            if len(val_series) > 0:
                iter_val_last.append(val_series[-1])
                iter_train_last.append(train_series[-1] if len(train_series) > 0 else float("nan"))
                iter_val_best.append(min(val_series))
                if iteration_index == iterations - 1:
                    last_iter_val_series = val_series
                    last_iter_train_series = train_series

            if print_rmse_each_iteration:
                self._quick_eval_rollout(quick_eval_episodes, max_episode_length)
                self._eval_rmse_by_dim_tables(
                    eval_seeds,
                    k_dim_list,
                    max_episode_length,
                    f"train iter {iteration_index}",
                    per_seed=False,
                )
                
                
            
        elapsed = int(time.time() - start_time)
        print(f"\nTraining finished. Elapsed: {elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}")
        self._print_hparam_recommendations(
            iter_val_last,
            iter_val_best,
            last_iter_train_series,
            last_iter_val_series,
        )

    def _print_hparam_recommendations(
        self,
        iter_val_last,
        iter_val_best,
        last_iter_train_series,
        last_iter_val_series,
    ):
        def _safe_ratio(a, b):
            if b == 0:
                return float("inf")
            return a / b

        # iterations: compare last vs best across iterations
        if len(iter_val_last) > 0:
            best_overall = min(iter_val_last)
            last_val = iter_val_last[-1]
            if last_val > best_overall * 1.05:
                iter_rec = "reduce"
            else:
                iter_rec = "increase"
        else:
            iter_rec = "increase"

        # train_epochs: use last iteration trend
        if len(last_iter_val_series) >= 2:
            min_val = min(last_iter_val_series)
            last_val = last_iter_val_series[-1]
            if last_val > min_val * 1.02:
                epoch_rec = "reduce"
            else:
                epoch_rec = "increase"
        else:
            epoch_rec = "increase"

        # batch_size: use val stability + train/val gap
        if len(last_iter_val_series) >= 2:
            mean_val = float(np.mean(last_iter_val_series))
            std_val = float(np.std(last_iter_val_series))
            cv = std_val / mean_val if mean_val > 0 else 0.0
        else:
            cv = 0.0

        if len(last_iter_train_series) > 0 and len(last_iter_val_series) > 0:
            gap_ratio = _safe_ratio(last_iter_train_series[-1], last_iter_val_series[-1])
        else:
            gap_ratio = 1.0

        if cv > 0.10 or gap_ratio < 0.7:
            batch_rec = "increase"
        else:
            batch_rec = "reduce"

        # hidden_sizes: underfit vs overfit from train/val gap
        if len(last_iter_train_series) > 0 and len(last_iter_val_series) > 0:
            gap_ratio = _safe_ratio(last_iter_train_series[-1], last_iter_val_series[-1])
            if gap_ratio > 0.9:
                hidden_rec = "increase"
            else:
                hidden_rec = "reduce"
        else:
            hidden_rec = "increase"

        # learning_rate: oscillation or rebound in val => reduce, otherwise increase
        if len(last_iter_val_series) >= 2:
            min_val = min(last_iter_val_series)
            last_val = last_iter_val_series[-1]
            mean_val = float(np.mean(last_iter_val_series))
            std_val = float(np.std(last_iter_val_series))
            cv = std_val / mean_val if mean_val > 0 else 0.0
            if last_val > min_val * 1.02 or cv > 0.10:
                lr_rec = "reduce"
            else:
                lr_rec = "increase"
        else:
            lr_rec = "increase"

        print("\n[hyperparam suggestions]")
        print(f"- iterations: {iter_rec}")
        print(f"- train_epochs: {epoch_rec}")
        print(f"- batch_size: {batch_rec}")
        print(f"- hidden_sizes: {hidden_rec}")
        print(f"- learning_rate: {lr_rec}")


    def evaluate(self):
        print("Overwriting base evaluate to predict model error")
        seeds = self.eval_config["seeds"]
        k_list = self.eval_config["k_list"]
        max_episode_length = int(self.train_config["max_episode_length"])
        max_k = max(k_list)
        k_dim_list = [k for k in [1, 5, 10, 15] if k <= max_k]
        if len(k_dim_list) == 0:
            k_dim_list = [1]

        eval_start_time = time.time()
        
        episode_rewards = []
        episode_forward_progresses = []

        # --- per-model accumulators (BASE vs BASE+RA) ---
        base_rmse_values_by_k = {k: [] for k in k_list}
        ra_rmse_values_by_k = {k: [] for k in k_list}

        base_sse_total_by_k = {k: None for k in k_dim_list}
        ra_sse_total_by_k = {k: None for k in k_dim_list}
        base_count_total_by_k = {k: 0 for k in k_dim_list}
        ra_count_total_by_k = {k: 0 for k in k_dim_list}

        dim_idx = 13
        k_targets = [k for k in [1, 5, 10, 15] if k in k_list]
        obs_dim = int(self.env.observation_space.shape[0])
        compute_dim_stats = obs_dim > dim_idx and len(k_targets) > 0
        base_dim_errs = {k: [] for k in k_targets}
        ra_dim_errs = {k: [] for k in k_targets}
        bin_labels = ["vx<0", "0<=vx<0.5", "0.5<=vx<1.0", "vx>=1.0"]
        base_dim_binned = {k: {label: [] for label in bin_labels} for k in k_targets}
        ra_dim_binned = {k: {label: [] for label in bin_labels} for k in k_targets}

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

            # ---------------- Dim 13 signed error diagnostics (accumulate) ----------------
            if compute_dim_stats:
                base_errs, base_bins = _k_step_dim_error_stats_for_episode(
                    episode_transitions, self.pretrained_dynamics_model, k_targets, self.device, dim_idx
                )
                ra_errs, ra_bins = _k_step_dim_error_stats_for_episode(
                    episode_transitions, self.residual_dynamics_wrapper, k_targets, self.device, dim_idx
                )
                for k in k_targets:
                    base_dim_errs[k].extend(base_errs.get(k, []))
                    ra_dim_errs[k].extend(ra_errs.get(k, []))
                    for label in bin_labels:
                        base_dim_binned[k][label].extend(base_bins.get(k, {}).get(label, []))
                        ra_dim_binned[k][label].extend(ra_bins.get(k, {}).get(label, []))

            # ---------------- Per-dim RMSE (k in k_dim_list) ----------------
            print()
            for k in k_dim_list:
                base_sse, base_count = compute_sse_by_dim_for_episode_k(
                    episode_transitions, self.pretrained_dynamics_model, k, self.device
                )
                ra_sse, ra_count = compute_sse_by_dim_for_episode_k(
                    episode_transitions, self.residual_dynamics_wrapper, k, self.device
                )

                if base_sse is not None and base_count > 0:
                    base_rmse_by_dim = torch.sqrt(base_sse / base_count).tolist()
                    if base_sse_total_by_k[k] is None:
                        base_sse_total_by_k[k] = base_sse.clone()
                    else:
                        base_sse_total_by_k[k] += base_sse
                    base_count_total_by_k[k] += base_count
                else:
                    base_rmse_by_dim = None

                if ra_sse is not None and ra_count > 0:
                    ra_rmse_by_dim = torch.sqrt(ra_sse / ra_count).tolist()
                    if ra_sse_total_by_k[k] is None:
                        ra_sse_total_by_k[k] = ra_sse.clone()
                    else:
                        ra_sse_total_by_k[k] += ra_sse
                    ra_count_total_by_k[k] += ra_count
                else:
                    ra_rmse_by_dim = None

                self._print_rmse_table(f"RMSE mean by dim (k-{k}) [seed {seed}]", base_rmse_by_dim, ra_rmse_by_dim)
                print()

        # ---------------- summary ----------------
        print("\n--------------------")

        base_mean_rmse_by_k = {k: float(np.mean(base_rmse_values_by_k[k])) for k in k_list}
        print("[BASE]    RMSE mean:", " | ".join([f"k-{k} {base_mean_rmse_by_k[k]:.4f}" for k in k_list]))
        ra_mean_rmse_by_k = {k: float(np.mean(ra_rmse_values_by_k[k])) for k in k_list}
        print("[BASE+RA] RMSE mean:", " | ".join([f"k-{k} {ra_mean_rmse_by_k[k]:.4f}" for k in k_list]))
        
        print()

        if not compute_dim_stats:
            if obs_dim <= dim_idx:
                print(f"Warning: obs dim {obs_dim} is too small for dim {dim_idx}; skipping v_x diagnostics.")
            else:
                print("Warning: no k in [1,5,10,15] found in k_list; skipping v_x diagnostics.")
            print()
        else:
            print("K-step error diagnostics (obs dim 13: torso x-velocity)")
            base_mean = {k: _summary_stats(base_dim_errs.get(k, []))[0] for k in k_targets}
            base_mae = {k: _summary_stats(base_dim_errs.get(k, []))[1] for k in k_targets}
            base_std = {k: _summary_stats(base_dim_errs.get(k, []))[2] for k in k_targets}
            ra_mean = {k: _summary_stats(ra_dim_errs.get(k, []))[0] for k in k_targets}
            ra_mae = {k: _summary_stats(ra_dim_errs.get(k, []))[1] for k in k_targets}
            ra_std = {k: _summary_stats(ra_dim_errs.get(k, []))[2] for k in k_targets}

            print("[BASE]    mean_err:", " | ".join([f"k-{k} {base_mean[k]:+.4f}" for k in k_targets]))
            print("[BASE]    MAE:", " | ".join([f"k-{k} {base_mae[k]:.4f}" for k in k_targets]))
            print("[BASE]    std_err:", " | ".join([f"k-{k} {base_std[k]:.4f}" for k in k_targets]))
            print("[BASE+RA] mean_err:", " | ".join([f"k-{k} {ra_mean[k]:+.4f}" for k in k_targets]))
            print("[BASE+RA] MAE:", " | ".join([f"k-{k} {ra_mae[k]:.4f}" for k in k_targets]))
            print("[BASE+RA] std_err:", " | ".join([f"k-{k} {ra_std[k]:.4f}" for k in k_targets]))

            print("Binned signed-error by true v_x (dim 13)")
            for label in bin_labels:
                base_stats = {k: _summary_stats(base_dim_binned[k][label]) for k in k_targets}
                ra_stats = {k: _summary_stats(ra_dim_binned[k][label]) for k in k_targets}
                print(
                    f"[BASE]    bin {label} mean_err:",
                    " | ".join([f"k-{k} {base_stats[k][0]:+.4f}" for k in k_targets]),
                )
                print(
                    f"[BASE]    bin {label} MAE:",
                    " | ".join([f"k-{k} {base_stats[k][1]:.4f}" for k in k_targets]),
                )
                print(
                    f"[BASE]    bin {label} std_err:",
                    " | ".join([f"k-{k} {base_stats[k][2]:.4f}" for k in k_targets]),
                )
                print(
                    f"[BASE+RA] bin {label} mean_err:",
                    " | ".join([f"k-{k} {ra_stats[k][0]:+.4f}" for k in k_targets]),
                )
                print(
                    f"[BASE+RA] bin {label} MAE:",
                    " | ".join([f"k-{k} {ra_stats[k][1]:.4f}" for k in k_targets]),
                )
                print(
                    f"[BASE+RA] bin {label} std_err:",
                    " | ".join([f"k-{k} {ra_stats[k][2]:.4f}" for k in k_targets]),
                )
            print()

        for k in k_dim_list:
            if base_sse_total_by_k[k] is not None and base_count_total_by_k[k] > 0:
                base_rmse_summary = torch.sqrt(
                    base_sse_total_by_k[k] / base_count_total_by_k[k]
                ).tolist()
            else:
                base_rmse_summary = None

            if ra_sse_total_by_k[k] is not None and ra_count_total_by_k[k] > 0:
                ra_rmse_summary = torch.sqrt(
                    ra_sse_total_by_k[k] / ra_count_total_by_k[k]
                ).tolist()
            else:
                ra_rmse_summary = None

            self._print_rmse_table(f"RMSE mean by dim (k-{k}) [all episodes]", base_rmse_summary, ra_rmse_summary)
            print()
        
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
   
