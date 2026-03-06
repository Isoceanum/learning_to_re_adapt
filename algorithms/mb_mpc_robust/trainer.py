import os

import numpy as np
from algorithms.base_trainer import BaseTrainer

import torch
import math
import time

from algorithms.mb_mpc_robust.dynamics_model import DynamicsModel
from algorithms.mb_mpc_robust.planner import RandomShootingPlanner, CrossEntropyMethodPlanner, MPPIPlanner
from algorithms.mb_mpc_robust.transition_buffer import TransitionBuffer


class MBMPCRobustTrainer(BaseTrainer):
    def __init__(self, config, output_dir):

        super().__init__(config, output_dir)
        
        self.env = self._make_train_env()
        self.dynamics_model = self._make_dynamics_model().to(self.device)
        self.planner =self._make_planner()
        self.buffer = self._make_buffer()
        self.action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        self.action_range = self.action_high - self.action_low
        self.exploration_config = self._make_exploration_config()
        self.exploration_rng = np.random.default_rng(self.train_seed + 12345)
        
    def _make_buffer(self):
        valid_split_ratio = float(self.train_config["valid_split_ratio"])
        return TransitionBuffer(valid_split_ratio, self.train_seed)
    
    def _make_dynamics_model(self):
        dynamics_model_config = self.train_config.get("dynamics_model")
        if dynamics_model_config is None: raise AttributeError("Missing dynamics_model config in YAML")
    
        observation_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        hidden_sizes = dynamics_model_config.get("hidden_sizes")
        learning_rate = float(dynamics_model_config.get("learning_rate"))
        seed = self.train_seed
        
        return DynamicsModel(observation_dim, action_dim, hidden_sizes, learning_rate, seed)

    def _make_planner(self):
        planner_config = self.train_config.get("planner")
        if planner_config is None:
            raise AttributeError("Missing planner config in YAML")
        
        planner_type = planner_config.get("type")         
        horizon = int(planner_config.get("horizon"))
        n_candidates = int(planner_config.get("n_candidates"))
        discount = float(planner_config.get("discount"))
        
        

        base_env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(base_env, "get_model_reward_fn"):
            raise AttributeError(f"Environment {self.env_id} does not implement get_model_reward_fn()")

        reward_fn = base_env.get_model_reward_fn()
        
        action_space = self.env.action_space
        act_low = action_space.low
        act_high = action_space.high
        seed = self.train_seed
        
        if planner_type == "rs":
            return RandomShootingPlanner(self.dynamics_model.predict_next_state, reward_fn, horizon, n_candidates, act_low, act_high, self.device, discount)
        
        if planner_type == "cem":
            num_cem_iters = int(planner_config.get("num_cem_iters"))
            percent_elites = float(planner_config.get("percent_elites"))
            alpha = float(planner_config.get("alpha"))        
            return CrossEntropyMethodPlanner(self.dynamics_model.predict_next_state, reward_fn, horizon, n_candidates, act_low, act_high, self.device, discount, num_cem_iters, percent_elites, alpha, seed)
        
        if planner_type == "mppi":
            noise_sigma = float(planner_config.get("noise_sigma"))
            lambda_ = float(planner_config.get("lambda_"))
            return MPPIPlanner(self.dynamics_model.predict_next_state, reward_fn, horizon, n_candidates, act_low, act_high, self.device, discount, noise_sigma, lambda_, seed)
            
            
        raise AttributeError(f"Planner type {planner_type} not supported")

    def _make_exploration_config(self):
        cfg = self.train_config.get("exploration", {})
        enabled = bool(cfg.get("enabled", False))
        explore_prob = float(cfg.get("explore_prob", 0.0))
        random_action_ratio = float(cfg.get("random_action_ratio", 1.0))
        noisy_action_std = float(cfg.get("noisy_action_std", 0.2))
        min_iteration = int(cfg.get("min_iteration", 1))

        if explore_prob < 0.0 or explore_prob > 1.0:
            raise ValueError(f"explore_prob must be in [0,1], got {explore_prob}")
        if random_action_ratio < 0.0 or random_action_ratio > 1.0:
            raise ValueError(f"random_action_ratio must be in [0,1], got {random_action_ratio}")
        if noisy_action_std < 0.0:
            raise ValueError(f"noisy_action_std must be >= 0, got {noisy_action_std}")
        if min_iteration < 1:
            raise ValueError(f"min_iteration must be >= 1, got {min_iteration}")

        return {
            "enabled": enabled,
            "explore_prob": explore_prob,
            "random_action_ratio": random_action_ratio,
            "noisy_action_std": noisy_action_std,
            "min_iteration": min_iteration,
        }

    def _planner_action(self, obs):
        action = self.planner.plan(obs)
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        return np.asarray(action, dtype=np.float32)

    def _noisy_planner_action(self, obs):
        base_action = self._planner_action(obs)
        noise_scale = self.exploration_config["noisy_action_std"] * self.action_range
        noise = self.exploration_rng.normal(loc=0.0, scale=noise_scale, size=base_action.shape)
        noisy_action = np.clip(base_action + noise, self.action_low, self.action_high)
        return noisy_action.astype(np.float32, copy=False)

    def _select_train_action(self, obs, iteration_index):
        if iteration_index == 0:
            action = np.asarray(self.env.action_space.sample(), dtype=np.float32)
            return action, "warmup_random"

        cfg = self.exploration_config
        if (not cfg["enabled"]) or iteration_index < cfg["min_iteration"]:
            return self._planner_action(obs), "planner"

        if self.exploration_rng.random() >= cfg["explore_prob"]:
            return self._planner_action(obs), "planner"

        if self.exploration_rng.random() < cfg["random_action_ratio"]:
            action = np.asarray(self.env.action_space.sample(), dtype=np.float32)
            return action, "explore_random"
        return self._noisy_planner_action(obs), "explore_noisy"

    def _predict_next_state_np(self, observation, action):
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred_next = self.dynamics_model.predict_next_state(obs_t, act_t)
        return np.asarray(pred_next.detach().cpu().numpy(), dtype=np.float32)

    def _resolve_dynamics_shift_eval_config(self):
        # Hard-coded robust diagnostics (kept out of YAML on purpose).
        enabled = True
        episodes = 3
        seeds = list(range(10))
        max_steps = 400
        max_horizon = 25
        policies = ["planner", "random", "noisy_planner"]
        noisy_planner_std = 0.20

        return {
            "enabled": enabled,
            "episodes": episodes,
            "seeds": seeds,
            "max_steps": max_steps,
            "max_horizon": max_horizon,
            "policies": policies,
            "noisy_planner_std": noisy_planner_std,
        }

    def _select_dynamics_eval_action(self, obs, policy_name, eval_env, noisy_planner_std, rng):
        if policy_name == "random":
            return np.asarray(eval_env.action_space.sample(), dtype=np.float32)
        if policy_name == "planner":
            return self._planner_action(obs)
        if policy_name == "noisy_planner":
            planner_action = self._planner_action(obs)
            noise_scale = noisy_planner_std * self.action_range
            noise = rng.normal(loc=0.0, scale=noise_scale, size=planner_action.shape)
            noisy_action = np.clip(planner_action + noise, self.action_low, self.action_high)
            return noisy_action.astype(np.float32, copy=False)
        raise ValueError(f"Unknown dynamics eval policy: {policy_name}")

    def _collect_dynamics_eval_rollout(self, eval_env, policy_name, max_steps, reset_seed, noisy_planner_std, rng):
        obs, _ = eval_env.reset(seed=reset_seed)
        obs = np.asarray(obs, dtype=np.float32)
        observations = [obs.copy()]
        actions = []

        for _ in range(max_steps):
            action = self._select_dynamics_eval_action(obs, policy_name, eval_env, noisy_planner_std, rng)
            next_obs, _, terminated, truncated, _ = eval_env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            actions.append(action.copy())
            observations.append(next_obs.copy())
            obs = next_obs
            if bool(terminated) or bool(truncated):
                break

        if len(actions) == 0:
            return None, None

        return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)

    def _init_dynamics_eval_accumulator(self, max_horizon):
        return {
            "one_step_sq_error_sum": 0.0,
            "one_step_count": 0,
            "sum_sq_by_h": np.zeros(max_horizon, dtype=np.float64),
            "count_by_h": np.zeros(max_horizon, dtype=np.int64),
            "rollouts": 0,
            "steps": 0,
        }

    def _accumulate_dynamics_eval_errors(self, acc, observations, actions, max_horizon):
        steps = int(actions.shape[0])
        if steps <= 0:
            return

        acc["rollouts"] += 1
        acc["steps"] += steps

        for start in range(steps):
            pred_next = self._predict_next_state_np(observations[start], actions[start])
            one_step_error = pred_next - observations[start + 1]
            acc["one_step_sq_error_sum"] += float(np.mean(one_step_error ** 2))
            acc["one_step_count"] += 1

            pred = observations[start]
            h_cap = min(max_horizon, steps - start)
            for h in range(1, h_cap + 1):
                pred = self._predict_next_state_np(pred, actions[start + h - 1])
                error = pred - observations[start + h]
                acc["sum_sq_by_h"][h - 1] += float(np.mean(error ** 2))
                acc["count_by_h"][h - 1] += 1

    def _finalize_dynamics_eval_errors(self, acc):
        one_step_rmse = float(
            np.sqrt(acc["one_step_sq_error_sum"] / max(1, acc["one_step_count"]))
        )
        counts = np.maximum(acc["count_by_h"], 1)
        rmse_by_h = np.sqrt(acc["sum_sq_by_h"] / counts)
        return one_step_rmse, rmse_by_h

    def _write_dynamics_shift_eval_results(self, config, results):
        out_path = os.path.join(self.output_dir, "dynamics_shift_eval.csv")
        k_candidates = [1, 5, 10, config["max_horizon"]]
        ks = []
        for k in k_candidates:
            if 1 <= k <= config["max_horizon"] and k not in ks:
                ks.append(k)

        with open(out_path, "w") as f:
            header_cols = ["policy", "one_step_rmse", "steps", "rollouts"] + [f"rmse_h{k}" for k in ks]
            f.write(",".join(header_cols) + "\n")
            for policy_name, result in results.items():
                row = [
                    policy_name,
                    f"{result['one_step_rmse']:.6f}",
                    str(result["steps"]),
                    str(result["rollouts"]),
                ]
                for k in ks:
                    row.append(f"{result['rmse_by_h'][k - 1]:.6f}")
                f.write(",".join(row) + "\n")

        print(f"Dynamics-shift eval CSV saved to {out_path}")

    def _evaluate_dynamics_shift(self):
        cfg = self._resolve_dynamics_shift_eval_config()
        if not cfg["enabled"]:
            return

        print(
            "\nDynamics-shift eval: "
            f"policies={cfg['policies']} "
            f"seeds={cfg['seeds']} "
            f"episodes={cfg['episodes']} "
            f"max_steps={cfg['max_steps']} "
            f"max_horizon={cfg['max_horizon']}"
        )

        rng = np.random.default_rng(self.train_seed + 67890)
        results = {}

        for policy_name in cfg["policies"]:
            acc = self._init_dynamics_eval_accumulator(cfg["max_horizon"])
            for seed in cfg["seeds"]:
                eval_env = self._make_eval_env(seed=seed)
                try:
                    for episode_index in range(cfg["episodes"]):
                        reset_seed = int(seed + episode_index * 1000)
                        observations, actions = self._collect_dynamics_eval_rollout(
                            eval_env=eval_env,
                            policy_name=policy_name,
                            max_steps=cfg["max_steps"],
                            reset_seed=reset_seed,
                            noisy_planner_std=cfg["noisy_planner_std"],
                            rng=rng,
                        )
                        if observations is None:
                            continue
                        self._accumulate_dynamics_eval_errors(
                            acc=acc,
                            observations=observations,
                            actions=actions,
                            max_horizon=cfg["max_horizon"],
                        )
                finally:
                    try:
                        eval_env.close()
                    except Exception:
                        pass

            one_step_rmse, rmse_by_h = self._finalize_dynamics_eval_errors(acc)
            results[policy_name] = {
                "one_step_rmse": one_step_rmse,
                "rmse_by_h": rmse_by_h,
                "steps": acc["steps"],
                "rollouts": acc["rollouts"],
            }
            h1 = rmse_by_h[0]
            h_last = rmse_by_h[cfg["max_horizon"] - 1]
            print(
                f"dynamics_eval[{policy_name}]: "
                f"one_step_rmse={one_step_rmse:.4f} "
                f"h1_rmse={h1:.4f} "
                f"h{cfg['max_horizon']}_rmse={h_last:.4f} "
                f"steps={acc['steps']} rollouts={acc['rollouts']}"
            )

        self._write_dynamics_shift_eval_results(cfg, results)
    
    def _collect_steps(self, iteration_index, steps_target, max_path_length):
        steps_collected_this_iteration = 0
            
        log_collect_start_time = time.time()
        log_episodes = 0
        log_episode_forward_progress = []
        log_episode_velocity = []
        log_episode_returns = []
        planner_steps = 0
        warmup_random_steps = 0
        explore_random_steps = 0
        explore_noisy_steps = 0
        
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
            
            while episode_steps < max_path_length:
                action, action_source = self._select_train_action(obs, iteration_index)
                if action_source == "planner":
                    planner_steps += 1
                elif action_source == "warmup_random":
                    warmup_random_steps += 1
                elif action_source == "explore_random":
                    explore_random_steps += 1
                elif action_source == "explore_noisy":
                    explore_noisy_steps += 1
                    
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
                
            self.buffer.add_trajectory(episode_obs, episode_act, episode_next_obs)
            log_episode_forward_progress.append(float(episode_x_last - episode_x_start))
            log_episode_velocity.append(float(episode_velocity))
            log_episode_returns.append(float(episode_return))
                    
        collect_stats = {
            "log_episodes": log_episodes,
            "log_collect_time":  time.time() - log_collect_start_time, 
            "steps_collected_this_iteration": steps_collected_this_iteration,
            "avg_reward": sum(log_episode_returns) / max(1, len(log_episode_returns)),
            "avg_forward_progress": sum(log_episode_forward_progress) / max(1, len(log_episode_forward_progress)),
            "avg_velocity": sum(log_episode_velocity) / max(1, len(log_episode_velocity)),
            "planner_steps": planner_steps,
            "warmup_random_steps": warmup_random_steps,
            "explore_random_steps": explore_random_steps,
            "explore_noisy_steps": explore_noisy_steps,
        }
        
        return collect_stats
    
    
    
    def _evaluate_dynamics_k_step(self):
        k_list=(1, 2, 5, 10, 15)
        k_max = max(k_list)
        
        episodes = self.buffer.eval_observations
        total_starts = sum(max(0, len(ep) - k_max + 1) for ep in episodes)
        eval_batch_size = min(5000, total_starts)
        
        obs_batch, action_batch, target_batch = self.buffer.sample_k_step_batch(k_max, eval_batch_size, "eval")
        obs_batch = obs_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        pred_state = obs_batch

    
        sum_squared_error_by_k = {k: 0.0 for k in k_list}
        count_by_k = {k: 0 for k in k_list}
        for t in range(k_max):
            act_t = action_batch[:, t, :]
            pred_next_state = self.dynamics_model.predict_next_state(pred_state, act_t)
            true_next_state = target_batch[:, t, :]
            error = pred_next_state - true_next_state
            step_index = t + 1
            if step_index in k_list:
                sum_squared_error_by_k[step_index] += (error ** 2).mean().item()
                count_by_k[step_index] += 1
                
            pred_state = pred_next_state
            
        mse_by_k = {k: (sum_squared_error_by_k[k] / max(1, count_by_k[k])) for k in k_list}
        rmse_by_k = {k: math.sqrt(mse_by_k[k]) for k in k_list}
        
        print("RMSE:", " | ".join([f"k-{k} {rmse_by_k[k]:.4f}" for k in k_list]))

    @torch.no_grad()
    def _log_delta_statistics(self, iteration_index: int):
        dm = self.dynamics_model
        if dm.mean_delta is None or dm.std_delta is None:
            return

        def summarize(values):
            mean = values.mean().item()
            std = values.std(unbiased=False).item()
            quantiles = torch.quantile(values, torch.tensor([0.9, 0.99], device=values.device))
            q90 = quantiles[0].item()
            q99 = quantiles[1].item()
            max_ = values.max().item()
            return mean, std, q90, q99, max_

        for split in ("train", "eval"):
            episodes = getattr(self.buffer, f"{split}_observations")
            num_episodes = len(episodes)
            num_transitions = sum(len(ep) for ep in episodes)
            print(f"[iter {iteration_index}] {split} episodes={num_episodes} transitions={num_transitions}")

            if num_transitions == 0:
                print(f"[iter {iteration_index}] no transitions available for split={split}")
                continue

            batch_size = min(num_transitions, 50_000)
            obs_batch, _, next_obs_batch = self.buffer.sample_transitions(batch_size, split)
            obs_batch = obs_batch.to(self.device)
            next_obs_batch = next_obs_batch.to(self.device)
            delta = next_obs_batch - obs_batch
            mean_delta = dm.mean_delta.to(self.device)
            std_delta = dm.std_delta.to(self.device)
            delta_norm = (delta - mean_delta) / std_delta

            raw_norm = torch.linalg.norm(delta, dim=-1)
            normalized_norm = torch.linalg.norm(delta_norm, dim=-1)

            raw_stats = summarize(raw_norm)
            norm_stats = summarize(normalized_norm)

            print(
                f"[iter {iteration_index}] delta_raw({split}): "
                f"mean={raw_stats[0]:.4f} std={raw_stats[1]:.4f} "
                f"p90={raw_stats[2]:.4f} p99={raw_stats[3]:.4f} max={raw_stats[4]:.4f}"
            )
            print(
                f"[iter {iteration_index}] delta_norm({split}): "
                f"mean={norm_stats[0]:.4f} std={norm_stats[1]:.4f} "
                f"p90={norm_stats[2]:.4f} p99={norm_stats[3]:.4f} max={norm_stats[4]:.4f}"
            )










    def _train_dynamics_for_iteration(self, train_epochs, batch_size, steps_per_epoch, eval_batch_size):
        log_print_every_k_epochs = 5
        rolling_p = 0.99
        eval_loss_ema = None
        eval_loss_ema_prev = None
        
        for _epoch in range(train_epochs):
            epoch_start_time = time.time()
            
            epoch_loss_sum = 0.0
            for _ in range(steps_per_epoch):
                batch_obs, batch_act, batch_next_obs = self.buffer.sample_transitions(batch_size, "train")
                
                batch_obs = batch_obs.to(self.device)
                batch_act = batch_act.to(self.device)
                batch_next_obs = batch_next_obs.to(self.device)
                
                loss_value = self.dynamics_model.train_on_batch(batch_obs, batch_act, batch_next_obs)
                epoch_loss_sum += loss_value
                    
            avg_epoch_loss = epoch_loss_sum / steps_per_epoch                    
            epoch_time_s = time.time() - epoch_start_time
            should_print = (_epoch % log_print_every_k_epochs == 0) or (_epoch == train_epochs - 1)
            
            
            # --- compute eval loss every epoch (needed for early stopping) ---
            eval_loss = float("nan")
            
            if eval_batch_size > 0 and steps_per_epoch > 0:
                eval_loss_sum = 0.0
                with torch.no_grad():
                    for _ in range(steps_per_epoch):
                        eval_obs_batch, eval_act_batch, eval_next_obs_batch = self.buffer.sample_transitions(eval_batch_size, "eval")
                        eval_obs_batch = eval_obs_batch.to(self.device)
                        eval_act_batch = eval_act_batch.to(self.device)
                        eval_next_obs_batch = eval_next_obs_batch.to(self.device)
                        eval_delta_batch = eval_next_obs_batch - eval_obs_batch
                        eval_loss_sum += self.dynamics_model.loss(eval_obs_batch, eval_act_batch, eval_delta_batch).item()
                eval_loss = eval_loss_sum / steps_per_epoch
                
            just_initialized_ema = False
            if eval_loss_ema is None and eval_batch_size > 0 and steps_per_epoch > 0:
                eval_loss_ema = 1.5 * eval_loss
                eval_loss_ema_prev = 2.0 * eval_loss
                just_initialized_ema = True
                
            if eval_loss_ema is not None:
                eval_loss_ema = rolling_p * eval_loss_ema + (1.0 - rolling_p) * eval_loss
                
            if (not just_initialized_ema) and (eval_loss_ema_prev is not None) and (eval_loss_ema_prev < eval_loss_ema):
                print(f"Early stopping at epoch {_epoch}: eval_ema worsened ({eval_loss_ema_prev:.6f} -> {eval_loss_ema:.6f})")
                print(f"epoch {_epoch}/{train_epochs}: " f"train={avg_epoch_loss:.6f} " f"eval={eval_loss:.6f} " f"time={epoch_time_s:.2f}s" )
                break
            
            eval_loss_ema_prev = eval_loss_ema
            
            if should_print:
                print(f"epoch {_epoch}/{train_epochs}: " f"train={avg_epoch_loss:.6f} " f"eval={eval_loss:.6f} " f"time={epoch_time_s:.2f}s" )
               
               
        self._evaluate_dynamics_k_step()
    
        
    def train(self):
        print("Starting MB-MPC training")
        start_time = time.time()
        
        max_path_length = int(self.train_config["max_path_length"])
        steps_per_iteration = int(self.train_config["steps_per_iteration"])
        iterations = int(self.train_config["iterations"])
        train_epochs = int(self.train_config["train_epochs"])
        batch_size = int(self.train_config["batch_size"]) 
        
        for iteration_index in range(iterations):
            print(f"\n ---------------- Iteration {iteration_index}/{iterations} ----------------")
            
            collect_stats = self._collect_steps(iteration_index, steps_per_iteration, max_path_length)
            num_train_transitions = sum(len(ep) for ep in self.buffer.train_observations)
            avg_reward = collect_stats["avg_reward"]
            avg_forward_progress = collect_stats["avg_forward_progress"]
            avg_velocity = collect_stats["avg_velocity"]
            steps_collected_this_iteration = collect_stats["steps_collected_this_iteration"]
            log_collect_time = collect_stats["log_collect_time"]
            log_episodes = collect_stats["log_episodes"]
            planner_steps = collect_stats["planner_steps"]
            warmup_random_steps = collect_stats["warmup_random_steps"]
            explore_random_steps = collect_stats["explore_random_steps"]
            explore_noisy_steps = collect_stats["explore_noisy_steps"]
            
            print(f"collect: dataset={num_train_transitions} " f"steps={steps_collected_this_iteration} " f"episodes={log_episodes} " f"avg_rew={avg_reward:.3f} " f"avg_fp={avg_forward_progress:.3f} " f"avg_v={avg_velocity:.3f} " f"time={log_collect_time:.1f}s")
            print(
                "collect_actions: "
                f"planner={planner_steps} "
                f"warmup_random={warmup_random_steps} "
                f"explore_random={explore_random_steps} "
                f"explore_noisy={explore_noisy_steps}"
            )
            
            mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta = self.buffer.get_normalization_stats()
            self.dynamics_model.update_normalization_stats(mean_obs, std_obs, mean_act, std_act, mean_delta, std_delta)
            self._log_delta_statistics(iteration_index)
      
            steps_per_epoch = math.ceil(num_train_transitions / batch_size)

            num_eval_transitions = sum(len(ep) for ep in self.buffer.eval_observations)
            eval_batch_size = min(num_eval_transitions, batch_size) if num_eval_transitions > 0 else 0
            self._train_dynamics_for_iteration(train_epochs, batch_size, steps_per_epoch, eval_batch_size)
            

        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        print(f"\nTraining finished. Elapsed: {h:02d}:{m:02d}:{s:02d}")
                                         
    def save(self):
        save_path = os.path.join(self.output_dir, "model.pt")
        
        self.dynamics_model._assert_normalization_stats()
        
        norm_stats = {
            "mean_obs": self.dynamics_model.mean_obs.detach().cpu(),
            "std_obs": self.dynamics_model.std_obs.detach().cpu(),
            "mean_act": self.dynamics_model.mean_act.detach().cpu(),
            "std_act": self.dynamics_model.std_act.detach().cpu(),
            "mean_delta": self.dynamics_model.mean_delta.detach().cpu(),
            "std_delta": self.dynamics_model.std_delta.detach().cpu(),
        }
   
        payload = {
            "state_dict": self.dynamics_model.state_dict(),
            "norm_stats": norm_stats,
        }
        torch.save(payload, save_path)
        print(f"Dynamics model saved to {save_path}")
        
    def predict(self, obs):
        return self._planner_action(obs)

    def _reset_eval_planner(self):
        self.planner = self._make_planner()

    def load(self, path):
        model_path = path
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")

        checkpoint = torch.load(model_path, map_location="cpu")

        # Restore model weights
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.dynamics_model.load_state_dict(state_dict)

        # Restore normalization stats required for planning
        normalization = checkpoint.get("norm_stats")
        if normalization is None:
            raise RuntimeError(
                "Checkpoint is missing normalization stats. Re-train with updated save() so stats are stored."
            )

        # Convert to tensors on correct device (update_normalization_stats handles numpy OR torch, but this is explicit)
        normalization = {
            k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in normalization.items()
        }

        self.dynamics_model.update_normalization_stats(
            normalization["mean_obs"], normalization["std_obs"],
            normalization["mean_act"], normalization["std_act"],
            normalization["mean_delta"], normalization["std_delta"],
        )

        print(f"Loaded dynamics model from {model_path}")
        return self
    
    def evaluate_checkpoint(self):
        dm = self.dynamics_model
        if any(v is None for v in (dm.mean_obs, dm.std_obs, dm.mean_act, dm.std_act, dm.mean_delta, dm.std_delta)):
            return
        super().evaluate_checkpoint()

    def evaluate(self):
        super().evaluate()
        self._evaluate_dynamics_shift()
