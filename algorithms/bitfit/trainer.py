import os
import time
import yaml
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base_trainer import BaseTrainer
from algorithms.bitfit.dynamics_model import DynamicsModel
from algorithms.bitfit.planner import CrossEntropyMethodPlanner, RandomShootingPlanner, MPPIPlanner
from algorithms.bitfit.transition_buffer import TransitionBuffer
from evaluation.model_error import compute_k_step_rmse_for_episode, compute_sse_by_dim_for_episode_k
from utils.seed import set_seed


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


class BitFitTrainer(BaseTrainer):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)
        self.env = self._make_train_env()
        self.base_dynamics_model = self._load_pretrained_dynamics_model(train_biases=False)
        self.bitfit_model = self._load_pretrained_dynamics_model(train_biases=True)
        self._cache_bias_init()
        self._log_param_status()
        self.optimizer = self._make_optimizer()
        self.planner = self._make_planner()
        self.base_planner = self._make_base_planner()
        self.grad_clip_recommendation = None

    def _make_dynamics_model(self, dynamics_model_config):
        hidden_sizes = dynamics_model_config["train"]["dynamics_model"]["hidden_sizes"]
        learning_rate = float(dynamics_model_config["train"]["dynamics_model"]["learning_rate"])
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        return DynamicsModel(obs_dim, action_dim, hidden_sizes, learning_rate, self.train_seed).to(self.device)

    def _load_pretrained_dynamics_model(self, train_biases: bool):
        model_path = self.train_config["pretrained_dynamics_model"]["model_path"]
        config_path = self.train_config["pretrained_dynamics_model"]["config_path"]

        with open(config_path, "r") as f:
            pretrained_config = yaml.safe_load(f)

        model = self._make_dynamics_model(pretrained_config)
        model.load_saved_model(model_path)

        if train_biases:
            self._enable_bias_training(model)
        else:
            model.freeze()

        return model

    def _enable_bias_training(self, model):
        for name, param in model.named_parameters():
            param.requires_grad_(name.endswith(".bias"))
        model.train()

    def _bias_parameters(self):
        return [(n, p) for n, p in self.bitfit_model.named_parameters() if n.endswith(".bias")]

    def _make_optimizer(self):
        bitfit_cfg = self.train_config.get("bitfit")
        if bitfit_cfg is None or "learning_rate" not in bitfit_cfg:
            raise AttributeError("Missing train.bitfit.learning_rate in YAML")
        learning_rate = float(bitfit_cfg["learning_rate"])

        bias_params = [p for _, p in self._bias_parameters()]
        if len(bias_params) == 0:
            raise RuntimeError("No bias parameters found for BitFit training.")

        return optim.Adam(bias_params, lr=learning_rate)

    def _cache_bias_init(self):
        self._bias_init = {}
        for name, param in self._bias_parameters():
            self._bias_init[name] = param.detach().clone()

    def _bias_delta_stats(self):
        if not hasattr(self, "_bias_init"):
            return float("nan"), float("nan")

        deltas = []
        abs_deltas = []
        for name, param in self._bias_parameters():
            init = self._bias_init.get(name)
            if init is None:
                continue
            delta = (param - init).reshape(-1)
            deltas.append(delta)
            abs_deltas.append(delta.abs())

        if not deltas:
            return float("nan"), float("nan")

        all_delta = torch.cat(deltas)
        all_abs = torch.cat(abs_deltas)
        delta_norm = float(torch.norm(all_delta).item())
        delta_mean_abs = float(torch.mean(all_abs).item())
        return delta_norm, delta_mean_abs

    def _log_param_status(self):
        total = sum(p.numel() for p in self.bitfit_model.parameters())
        trainable = sum(p.numel() for p in self.bitfit_model.parameters() if p.requires_grad)
        frozen = total - trainable
        bias_params = sum(p.numel() for _, p in self._bias_parameters())
        print(f"[bitfit params] total={total} trainable={trainable} frozen={frozen} bias_params={bias_params}")

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
            return CrossEntropyMethodPlanner(
                dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount, num_cem_iters, percent_elites, alpha, seed
            )

        if planner_type == "rs":
            return RandomShootingPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, self.device, discount)

        if planner_type == "mppi":
            noise_sigma = float(planner_config.get("noise_sigma"))
            lambda_ = float(planner_config.get("lambda_"))
            return MPPIPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, self.device, discount, noise_sigma, lambda_, seed)

        raise AttributeError(f"Planner type {planner_type} not supported")

    def _make_planner(self):
        planner_config = self.train_config.get("planner")
        action_space = self.env.action_space

        base_env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(base_env, "get_model_reward_fn"):
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")

        reward_fn = base_env.get_model_reward_fn()
        dynamics_fn = self.bitfit_model.predict_next_state
        return self.make_planner(planner_config, dynamics_fn, reward_fn, action_space, self.device, self.train_seed)

    def _make_base_planner(self):
        planner_config = self.train_config.get("planner")
        action_space = self.env.action_space

        base_env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(base_env, "get_model_reward_fn"):
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")

        reward_fn = base_env.get_model_reward_fn()
        return self.make_planner(planner_config, self.base_dynamics_model.predict_next_state, reward_fn, action_space, self.device, self.train_seed)

    def _collect_env_steps(self, steps_target, max_episode_length, use_base_only=False):
        collect_start_time = time.time()

        steps_collected_this_iteration = 0
        log_episodes = 0
        log_episode_returns = []
        log_episode_forward_progress = []

        buffer = TransitionBuffer(
            valid_split_ratio=float(self.train_config["valid_split_ratio"]),
            seed=self.train_seed,
        )

        while steps_collected_this_iteration < steps_target:
            obs, _ = self.env.reset()
            log_episodes += 1

            episode_return = 0.0
            episode_x_start = None
            episode_x_last = None
            episode_steps = 0

            episode_obs = []
            episode_act = []
            episode_next_obs = []

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

            buffer.add_trajectory(episode_obs, episode_act, episode_next_obs)
            log_episode_forward_progress.append(float(episode_x_last - episode_x_start) if (episode_x_start is not None and episode_x_last is not None) else 0.0)
            log_episode_returns.append(float(episode_return))

        log_collect_time = time.time() - collect_start_time
        reward_mean = float(np.mean(log_episode_returns)) if log_episode_returns else 0.0
        reward_std = float(np.std(log_episode_returns)) if log_episode_returns else 0.0

        forward_mean = float(np.mean(log_episode_forward_progress)) if log_episode_forward_progress else 0.0
        forward_std = float(np.std(log_episode_forward_progress)) if log_episode_forward_progress else 0.0

        print(
            f"Collected: steps={steps_collected_this_iteration} "
            f"episodes={log_episodes} "
            f"reward_mean={reward_mean:.3f} ± {reward_std:.3f} "
            f"forward_mean={forward_mean:.3f} ± {forward_std:.3f} "
            f"time={log_collect_time:.1f}s"
        )

        return buffer

    def _make_dataloaders(self, buffer: TransitionBuffer):
        train_obs, train_act, train_next_obs, val_obs, val_act, val_next_obs = buffer.get_split_arrays()

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

    def _run_epoch(self, loader, train):
        if self.optimizer is None and train:
            raise RuntimeError("Optimizer is not initialized.")

        stats = {
            "loss": [],
            "base_mse": [],
            "pred_mse": [],
        }

        base = self.base_dynamics_model
        model = self.bitfit_model
        mean_obs = model.mean_obs
        std_obs = model.std_obs

        def _accumulate(loss, base_mse, pred_mse):
            stats["loss"].append(float(loss))
            stats["base_mse"].append(float(base_mse))
            stats["pred_mse"].append(float(pred_mse))

        if train:
            model.train()
            for obs_b, act_b, next_obs_b in loader:
                self.optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    base_pred_next = base.predict_next_state(obs_b, act_b)

                pred_next = model.predict_next_state(obs_b, act_b)

                next_obs_norm = (next_obs_b - mean_obs) / std_obs
                pred_next_norm = (pred_next - mean_obs) / std_obs
                base_pred_next_norm = (base_pred_next - mean_obs) / std_obs

                loss = torch.mean((pred_next_norm - next_obs_norm) ** 2)
                loss.backward()
                self.optimizer.step()

                base_mse = torch.mean((base_pred_next_norm - next_obs_norm) ** 2)
                pred_mse = torch.mean((pred_next_norm - next_obs_norm) ** 2)

                _accumulate(loss.item(), base_mse.item(), pred_mse.item())
        else:
            model.eval()
            with torch.no_grad():
                for obs_b, act_b, next_obs_b in loader:
                    base_pred_next = base.predict_next_state(obs_b, act_b)
                    pred_next = model.predict_next_state(obs_b, act_b)

                    next_obs_norm = (next_obs_b - mean_obs) / std_obs
                    pred_next_norm = (pred_next - mean_obs) / std_obs
                    base_pred_next_norm = (base_pred_next - mean_obs) / std_obs

                    loss = torch.mean((pred_next_norm - next_obs_norm) ** 2)
                    base_mse = torch.mean((base_pred_next_norm - next_obs_norm) ** 2)
                    pred_mse = torch.mean((pred_next_norm - next_obs_norm) ** 2)

                    _accumulate(loss.item(), base_mse.item(), pred_mse.item())

        if len(stats["loss"]) == 0:
            return {k: float("nan") for k in stats}

        return {k: float(np.mean(v)) for k, v in stats.items()}

    def train(self):
        print("Starting BitFit Phase-1 training")
        start_time = time.time()

        saw_nonfinite_loss = False
        print_rmse_each_iteration = False

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
                print("Collecting rollout data with bitfit planner")

            buffer = self._collect_env_steps(
                steps_per_iteration,
                max_episode_length,
                use_base_only=use_base_only,
            )
            train_loader, val_loader = self._make_dataloaders(buffer)

            patience = 2
            min_delta = 1e-4
            bad_epochs = 0
            best_val = float("inf")
            best_epoch = 0
            best_state = {name: param.detach().clone() for name, param in self._bias_parameters()}
            early_stop = False
            stopped_epoch = train_epochs
            val_series = []
            train_series = []
            for epoch in range(train_epochs):
                train_stats = self._run_epoch(train_loader, train=True)
                val_stats = self._run_epoch(val_loader, train=False)
                train_series.append(float(train_stats["pred_mse"]))
                val_series.append(float(val_stats["pred_mse"]))
                if (not np.isfinite(train_stats["pred_mse"])) or (not np.isfinite(val_stats["pred_mse"])):
                    saw_nonfinite_loss = True

                current_val = float(val_stats["pred_mse"])
                if np.isfinite(current_val):
                    if best_val - current_val >= min_delta:
                        best_val = current_val
                        best_epoch = epoch + 1
                        best_state = {name: param.detach().clone() for name, param in self._bias_parameters()}
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                else:
                    bad_epochs += 1

                bias_delta_norm, bias_delta_mean_abs = self._bias_delta_stats()

                print(
                    f"epoch {epoch+1}/{train_epochs} "
                    f"train_base_mse={train_stats['base_mse']:.6f} "
                    f"train_pred_mse={train_stats['pred_mse']:.6f} "
                    f"val_base_mse={val_stats['base_mse']:.6f} "
                    f"val_pred_mse={val_stats['pred_mse']:.6f} "
                    f"bias_delta_norm={bias_delta_norm:.6f} "
                    f"bias_delta_mean_abs={bias_delta_mean_abs:.6f}"
                )

                if bad_epochs >= patience:
                    early_stop = True
                    stopped_epoch = epoch + 1
                    break

            if best_state:
                for name, param in self._bias_parameters():
                    if name in best_state:
                        param.data.copy_(best_state[name])

            best_val_print = best_val if np.isfinite(best_val) else float("nan")
            print(
                "early_stop="
                f"{early_stop} "
                f"best_epoch={best_epoch} "
                f"best_val_pred_mse={best_val_print:.6f} "
                f"stopped_epoch={stopped_epoch}"
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
        self._set_grad_clip_recommendation(last_iter_train_series, saw_nonfinite_loss)
        msg = self.grad_clip_recommendation_message()
        if msg:
            print(msg)

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

        if len(iter_val_last) > 0:
            best_overall = min(iter_val_last)
            last_val = iter_val_last[-1]
            if last_val > best_overall * 1.05:
                iter_rec = "reduce"
            else:
                iter_rec = "increase"
        else:
            iter_rec = "increase"

        if len(last_iter_val_series) >= 2:
            min_val = min(last_iter_val_series)
            last_val = last_iter_val_series[-1]
            if last_val > min_val * 1.02:
                epoch_rec = "reduce"
            else:
                epoch_rec = "increase"
        else:
            epoch_rec = "increase"

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

        if len(last_iter_train_series) > 0 and len(last_iter_val_series) > 0:
            gap_ratio = _safe_ratio(last_iter_train_series[-1], last_iter_val_series[-1])
            if gap_ratio > 0.9:
                hidden_rec = "increase"
            else:
                hidden_rec = "reduce"
        else:
            hidden_rec = "increase"

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

    def _set_grad_clip_recommendation(self, last_iter_train_series, saw_nonfinite_loss):
        if saw_nonfinite_loss:
            self.grad_clip_recommendation = {
                "should_clip": True,
                "reason": "nonfinite_loss_detected",
            }
            return

        should_clip = False
        reason = "stable_loss"
        if len(last_iter_train_series) >= 3:
            mean_val = float(np.mean(last_iter_train_series))
            std_val = float(np.std(last_iter_train_series))
            cv = std_val / mean_val if mean_val > 0 else 0.0
            min_val = min(last_iter_train_series)
            max_val = max(last_iter_train_series)
            ratio = (max_val / min_val) if min_val > 0 else float("inf")
            if cv > 0.5 or ratio > 5.0:
                should_clip = True
                reason = "loss_volatility"
        else:
            reason = "insufficient_history"

        self.grad_clip_recommendation = {
            "should_clip": should_clip,
            "reason": reason,
        }

    def grad_clip_recommendation_message(self):
        if self.grad_clip_recommendation is None:
            return None
        should_clip = self.grad_clip_recommendation.get("should_clip", False)
        return f"[grad_clip_norm] recommendation: {'implement' if should_clip else 'do not implement'}"

    def _reset_eval_planner(self):
        self.planner = self._make_planner()
        self.base_planner = self._make_base_planner()

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
        print(
            f"[quick-eval] episodes={episodes} reward_mean={reward_mean:.3f} ± {reward_std:.3f} "
            f"forward_mean={forward_mean:.3f} ± {forward_std:.3f}"
        )

    def _print_rmse_table(self, title, base_rmse, bf_rmse, label_width=10, num_width=7):
        print(title)
        if base_rmse is None or bf_rmse is None:
            print(f"{'BASE':<{label_width}} n/a")
            print(f"{'BASE+BF':<{label_width}} n/a")
            return

        base_list = [float(v) for v in base_rmse]
        bf_list = [float(v) for v in bf_rmse]
        num_dims = min(len(base_list), len(bf_list))
        if num_dims == 0:
            print(f"{'BASE':<{label_width}} n/a")
            print(f"{'BASE+BF':<{label_width}} n/a")
            return

        dims = range(0, num_dims)
        header = f"{'DIM.':<{label_width}}" + "".join([f"{d:>{num_width}d}" for d in dims])
        base_line = f"{'BASE':<{label_width}}" + "".join([f"{base_list[d]:>{num_width}.3f}" for d in dims])
        bf_line = f"{'BASE+BF':<{label_width}}" + "".join([f"{bf_list[d]:>{num_width}.3f}" for d in dims])
        print(header)
        print(base_line)
        print(bf_line)

    def _eval_rmse_by_dim_tables(self, seeds, k_dim_list, max_episode_length, header_prefix, per_seed=True):
        if not k_dim_list:
            return

        base_sse_total_by_k = {k: None for k in k_dim_list}
        bf_sse_total_by_k = {k: None for k in k_dim_list}
        base_count_total_by_k = {k: 0 for k in k_dim_list}
        bf_count_total_by_k = {k: 0 for k in k_dim_list}

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
                    episode_transitions, self.base_dynamics_model, k, self.device
                )
                bf_sse, bf_count = compute_sse_by_dim_for_episode_k(
                    episode_transitions, self.bitfit_model, k, self.device
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

                if bf_sse is not None and bf_count > 0:
                    bf_rmse_by_dim = torch.sqrt(bf_sse / bf_count).tolist()
                    if bf_sse_total_by_k[k] is None:
                        bf_sse_total_by_k[k] = bf_sse.clone()
                    else:
                        bf_sse_total_by_k[k] += bf_sse
                    bf_count_total_by_k[k] += bf_count
                else:
                    bf_rmse_by_dim = None

                if per_seed:
                    self._print_rmse_table(
                        f"RMSE mean by dim (k-{k}) [{header_prefix} seed {seed}]",
                        base_rmse_by_dim,
                        bf_rmse_by_dim,
                    )
                    print()

        for k in k_dim_list:
            if base_sse_total_by_k[k] is not None and base_count_total_by_k[k] > 0:
                base_rmse_summary = torch.sqrt(base_sse_total_by_k[k] / base_count_total_by_k[k]).tolist()
            else:
                base_rmse_summary = None

            if bf_sse_total_by_k[k] is not None and bf_count_total_by_k[k] > 0:
                bf_rmse_summary = torch.sqrt(bf_sse_total_by_k[k] / bf_count_total_by_k[k]).tolist()
            else:
                bf_rmse_summary = None

            self._print_rmse_table(
                f"RMSE mean by dim (k-{k}) [{header_prefix} all episodes]",
                base_rmse_summary,
                bf_rmse_summary,
            )
            print()

    def evaluate(self):
        print("Evaluating BitFit: base vs base+bitfit")
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

        base_rmse_values_by_k = {k: [] for k in k_list}
        bf_rmse_values_by_k = {k: [] for k in k_list}

        base_sse_total_by_k = {k: None for k in k_dim_list}
        bf_sse_total_by_k = {k: None for k in k_dim_list}
        base_count_total_by_k = {k: 0 for k in k_dim_list}
        bf_count_total_by_k = {k: 0 for k in k_dim_list}

        bitfit_eval_cfg = self.eval_config.get("bitfit")
        if bitfit_eval_cfg is None or "dim_idx" not in bitfit_eval_cfg:
            raise AttributeError("Missing eval.bitfit.dim_idx in YAML")
        dim_idx = int(bitfit_eval_cfg["dim_idx"])
        k_targets = [k for k in [1, 5, 10, 15] if k in k_list]
        obs_dim = int(self.env.observation_space.shape[0])
        compute_dim_stats = obs_dim > dim_idx and len(k_targets) > 0
        base_dim_errs = {k: [] for k in k_targets}
        bf_dim_errs = {k: [] for k in k_targets}
        bin_labels = ["vx<0", "0<=vx<0.5", "0.5<=vx<1.0", "vx>=1.0"]
        base_dim_binned = {k: {label: [] for label in bin_labels} for k in k_targets}
        bf_dim_binned = {k: {label: [] for label in bin_labels} for k in k_targets}

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

            print()
            base_rmse_by_k = compute_k_step_rmse_for_episode(episode_transitions, self.base_dynamics_model, k_list, self.device)
            for k in k_list:
                base_rmse_values_by_k[k].append(base_rmse_by_k[k])

            print("[BASE]    RMSE:", " | ".join([f"k-{k} {base_rmse_by_k[k]:.4f}" for k in k_list]))

            bf_rmse_by_k = compute_k_step_rmse_for_episode(episode_transitions, self.bitfit_model, k_list, self.device)
            for k in k_list:
                bf_rmse_values_by_k[k].append(bf_rmse_by_k[k])

            print("[BASE+BF] RMSE:", " | ".join([f"k-{k} {bf_rmse_by_k[k]:.4f}" for k in k_list]))

            if compute_dim_stats:
                base_errs, base_bins = _k_step_dim_error_stats_for_episode(
                    episode_transitions, self.base_dynamics_model, k_targets, self.device, dim_idx
                )
                bf_errs, bf_bins = _k_step_dim_error_stats_for_episode(
                    episode_transitions, self.bitfit_model, k_targets, self.device, dim_idx
                )
                for k in k_targets:
                    base_dim_errs[k].extend(base_errs.get(k, []))
                    bf_dim_errs[k].extend(bf_errs.get(k, []))
                    for label in bin_labels:
                        base_dim_binned[k][label].extend(base_bins.get(k, {}).get(label, []))
                        bf_dim_binned[k][label].extend(bf_bins.get(k, {}).get(label, []))

            print()
            for k in k_dim_list:
                base_sse, base_count = compute_sse_by_dim_for_episode_k(
                    episode_transitions, self.base_dynamics_model, k, self.device
                )
                bf_sse, bf_count = compute_sse_by_dim_for_episode_k(
                    episode_transitions, self.bitfit_model, k, self.device
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

                if bf_sse is not None and bf_count > 0:
                    bf_rmse_by_dim = torch.sqrt(bf_sse / bf_count).tolist()
                    if bf_sse_total_by_k[k] is None:
                        bf_sse_total_by_k[k] = bf_sse.clone()
                    else:
                        bf_sse_total_by_k[k] += bf_sse
                    bf_count_total_by_k[k] += bf_count
                else:
                    bf_rmse_by_dim = None

                self._print_rmse_table(f"RMSE mean by dim (k-{k}) [seed {seed}]", base_rmse_by_dim, bf_rmse_by_dim)
                print()

        print("\n--------------------")
        base_mean_rmse_by_k = {k: float(np.mean(base_rmse_values_by_k[k])) for k in k_list}
        print("[BASE]    RMSE mean:", " | ".join([f"k-{k} {base_mean_rmse_by_k[k]:.4f}" for k in k_list]))
        bf_mean_rmse_by_k = {k: float(np.mean(bf_rmse_values_by_k[k])) for k in k_list}
        print("[BASE+BF] RMSE mean:", " | ".join([f"k-{k} {bf_mean_rmse_by_k[k]:.4f}" for k in k_list]))

        print()

        if not compute_dim_stats:
            if obs_dim <= dim_idx:
                print(f"Warning: obs dim {obs_dim} is too small for dim {dim_idx}; skipping v_x diagnostics.")
            else:
                print("Warning: no k in [1,5,10,15] found in k_list; skipping v_x diagnostics.")
            print()
        else:
            print("K-step error diagnostics (obs dim for v_x)")
            base_mean = {k: _summary_stats(base_dim_errs.get(k, []))[0] for k in k_targets}
            base_mae = {k: _summary_stats(base_dim_errs.get(k, []))[1] for k in k_targets}
            base_std = {k: _summary_stats(base_dim_errs.get(k, []))[2] for k in k_targets}
            bf_mean = {k: _summary_stats(bf_dim_errs.get(k, []))[0] for k in k_targets}
            bf_mae = {k: _summary_stats(bf_dim_errs.get(k, []))[1] for k in k_targets}
            bf_std = {k: _summary_stats(bf_dim_errs.get(k, []))[2] for k in k_targets}

            print("[BASE]    mean_err:", " | ".join([f"k-{k} {base_mean[k]:+.4f}" for k in k_targets]))
            print("[BASE]    MAE:", " | ".join([f"k-{k} {base_mae[k]:.4f}" for k in k_targets]))
            print("[BASE]    std_err:", " | ".join([f"k-{k} {base_std[k]:.4f}" for k in k_targets]))
            print("[BASE+BF] mean_err:", " | ".join([f"k-{k} {bf_mean[k]:+.4f}" for k in k_targets]))
            print("[BASE+BF] MAE:", " | ".join([f"k-{k} {bf_mae[k]:.4f}" for k in k_targets]))
            print("[BASE+BF] std_err:", " | ".join([f"k-{k} {bf_std[k]:.4f}" for k in k_targets]))

            print("Binned signed-error by true v_x (dim index)")
            for label in bin_labels:
                base_stats = {k: _summary_stats(base_dim_binned[k][label]) for k in k_targets}
                bf_stats = {k: _summary_stats(bf_dim_binned[k][label]) for k in k_targets}
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
                    f"[BASE+BF] bin {label} mean_err:",
                    " | ".join([f"k-{k} {bf_stats[k][0]:+.4f}" for k in k_targets]),
                )
                print(
                    f"[BASE+BF] bin {label} MAE:",
                    " | ".join([f"k-{k} {bf_stats[k][1]:.4f}" for k in k_targets]),
                )
                print(
                    f"[BASE+BF] bin {label} std_err:",
                    " | ".join([f"k-{k} {bf_stats[k][2]:.4f}" for k in k_targets]),
                )
            print()

        for k in k_dim_list:
            if base_sse_total_by_k[k] is not None and base_count_total_by_k[k] > 0:
                base_rmse_summary = torch.sqrt(base_sse_total_by_k[k] / base_count_total_by_k[k]).tolist()
            else:
                base_rmse_summary = None

            if bf_sse_total_by_k[k] is not None and bf_count_total_by_k[k] > 0:
                bf_rmse_summary = torch.sqrt(bf_sse_total_by_k[k] / bf_count_total_by_k[k]).tolist()
            else:
                bf_rmse_summary = None

            self._print_rmse_table(f"RMSE mean by dim (k-{k}) [all episodes]", base_rmse_summary, bf_rmse_summary)
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
        save_path = os.path.join(self.output_dir, "bitfit_model.pt")
        payload = {
            "bitfit_model_state": self.bitfit_model.state_dict(),
        }
        if self.optimizer is not None:
            payload["optimizer_state"] = self.optimizer.state_dict()
        torch.save(payload, save_path)
        print(f"BitFit model saved to {save_path}")

    def load(self, path):
        model_path = path
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "bitfit_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get("bitfit_model_state", checkpoint)
        self.bitfit_model.load_state_dict(state_dict)
        self._enable_bias_training(self.bitfit_model)

        if self.optimizer is not None and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        print(f"Loaded BitFit model from {model_path}")
        return self
