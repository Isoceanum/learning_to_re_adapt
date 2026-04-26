import csv
import os
from perturbations.perturbation_factory import resolve_perturbation_env, wrap_perturbation
from utils.seed import seed_env, set_seed
import time
import numpy as np
import envs, gymnasium as gym
import torch

class BaseTrainer:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.environment_config = config["environment"]
        self.train_config = config["train"]
        self.eval_config = config["eval"]
        self.deterministic_evaluation = bool(self.eval_config.get("deterministic_evaluation", False))
        self.env_id = self.environment_config["id"]
        self.device = self._resolve_device()
        print("Using device : ", self.device)
        self.train_seed = self.train_config["seed"]
        train_perturbation_config = self.train_config.get("perturbation", {})
        self.env = self._make_env(self.environment_config, train_perturbation_config, self.train_seed)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.total_steps = 0
        self.eval_interval_steps = self.train_config.get("eval_interval_steps")
        self.next_eval_steps = int(self.eval_interval_steps) if self.eval_interval_steps is not None else None
        self.current_task = None

        
    def _make_env(self, environment_config, perturbation_config, seed):
        env_id = environment_config["id"]
        env_kwargs = {}
        env_kwargs["exclude_current_positions_from_observation"] = environment_config["exclude_current_positions_from_observation"]

        env = gym.make(env_id, **env_kwargs)
        env = wrap_perturbation(env, perturbation_config, seed)
        env.reset(seed=seed)
        seed_env(env, seed)
        return env

    def _get_task_from_env(self, env):
        get_task = getattr(env, "get_task", None)
        if callable(get_task):
            return str(get_task())
        return "nominal"

    def _reset_planner_rng_for_eval(self, seed):
        if not self.deterministic_evaluation:
            return
        planner = getattr(self, "planner", None)
        if planner is None:
            return
        generator = getattr(planner, "gen", None)
        if generator is None:
            return
        generator.manual_seed(int(seed))
  
    def _evaluate(self, episodes, seeds):
        all_rewards = []
        episode_lengths = []
        eval_start_time = time.time()
        eval_perturbation_config = self.eval_config.get("perturbation", {})
        max_episode_length = int(self.environment_config["max_episode_length"])

        for seed in seeds:
            for episode in range(episodes):
                # Deterministic per-episode evaluation:
                # reseed global RNGs, reset planner RNG, and recreate eval env.
                set_seed(seed)
                self._reset_planner_rng_for_eval(seed)
                eval_env = self._make_env(self.environment_config, eval_perturbation_config, seed)
                trajectory, metrics = self._rollout_episode(eval_env, 1, max_episode_length, reset_seed=seed)
                episode_obs, _, _ = trajectory
                all_rewards.append(metrics["episode_return"])
                episode_lengths.append(int(len(episode_obs)))
                self._reset_episode_state()
                eval_env.close()
        
        return {
            "reward_mean": np.mean(all_rewards),
            "reward_std": np.std(all_rewards),
            "episode_length_mean": np.mean(episode_lengths),
            "elapsed": time.time() - eval_start_time,
        }

    def evaluate(self):
        episodes = int(self.eval_config["episodes"])
        seeds = self.eval_config["seeds"]
        
        total_runs = len(seeds) * episodes
        print(f"Evaluating {episodes} episode(s) × {len(seeds)} seed(s) = {total_runs} total runs")
        
        metrics = self._evaluate(episodes, seeds)
        elapsed = metrics["elapsed"]
        elapsed_str = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"

        print("\nEvaluation summary:")
        print(f"- reward_mean: {metrics['reward_mean']:.4f}")
        print(f"- reward_std: {metrics['reward_std']:.4f}")
        print(f"- episode_length_mean: {metrics['episode_length_mean']:.2f}")
        print(f"- elapsed: {elapsed_str}")
         
    def _rollout_episode(self, env, iteration_index, max_steps, reset_seed=None):
        if reset_seed is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(seed=int(reset_seed))
        self.current_task = self._get_task_from_env(env)
        episode_return = 0.0
        episode_steps = 0
        episode_obs = []
        episode_act = []
        episode_next_obs = []

        while episode_steps < max_steps:
            if iteration_index == 0:
                action = env.action_space.sample()
            else:
                action = self.predict(obs)
                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)

            episode_obs.append(obs)
            episode_act.append(action)
            episode_next_obs.append(next_obs)

            obs = next_obs
            episode_steps += 1

            if terminated or truncated:
                break
            
        return (episode_obs, episode_act, episode_next_obs), {"episode_return": float(episode_return)}

    def collect_steps(self, iteration_index, steps_target):
        if not hasattr(self, "buffer") or self.buffer is None:
            raise RuntimeError("buffer not set")

        max_path_length = int(self.environment_config["max_episode_length"])
        steps_collected_this_iteration = 0
        log_collect_start_time = time.time()
        log_episodes = 0

        log_episode_returns = []
        log_episode_lengths = []

        while steps_collected_this_iteration < steps_target:
            steps_remaining = steps_target - steps_collected_this_iteration
            episode_max_steps = min(max_path_length, steps_remaining)
            trajectory, metrics = self._rollout_episode(self.env, iteration_index, episode_max_steps)
            episode_obs, episode_act, episode_next_obs = trajectory

            self.buffer.add_trajectory(episode_obs, episode_act, episode_next_obs)
            log_episode_returns.append(metrics["episode_return"])
            log_episode_lengths.append(int(len(episode_obs)))

            steps_collected_this_iteration += len(episode_obs)
            log_episodes += 1
            self._reset_episode_state()

        reward_mean = float(np.mean(log_episode_returns)) if log_episode_returns else 0.0
        reward_std = float(np.std(log_episode_returns)) if log_episode_returns else 0.0
        avg_episode_length = float(np.mean(log_episode_lengths)) if log_episode_lengths else 0.0

        self.total_steps += steps_collected_this_iteration
        if self.eval_interval_steps is not None and self.total_steps >= self.next_eval_steps:
            self._checkpoint_eval()
            self.next_eval_steps += self.eval_interval_steps

        log_collect_time = time.time() - log_collect_start_time
        print(
            f"collect: steps={steps_collected_this_iteration} "
            f"episodes={log_episodes} "
            f"avg_rew={reward_mean:.3f} "
            f"rew_std={reward_std:.3f} "
            f"avg_ep_len={avg_episode_length:.1f} "
            f"time={log_collect_time:.1f}s"
        )
        self._log_collect_csv(steps_collected_this_iteration, reward_mean, reward_std, log_collect_time)

    def _log_collect_csv(self, steps_collected, reward_mean, reward_std, time, filename="progress.csv"):
        path = os.path.join(self.output_dir, filename)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["steps", "avg_reward", "reward_std", "time"])

            writer.writerow([int(steps_collected), f"{reward_mean:.3f}", f"{reward_std:.3f}",f"{time:.2f}"])
          
    def _resolve_device(self):
        device = self.config["device"].lower()
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        
        if device == "auto":
            return torch.device("cuda" if cuda_available else "cpu")
        
        if device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA requested but is not available")
    
        return torch.device(device)
        
    def train(self):
       raise NotImplementedError("train() must be implemented in subclass")

    def predict(self, obs):
        raise NotImplementedError("predict() must be implemented in subclass")
    
    def load(self, path):
        raise NotImplementedError("load() must be implemented in subclass")
    
    def save(self):
        raise NotImplementedError("save() must be implemented in subclass")
    
    def _reset_episode_state(self):
        return

    def _checkpoint_eval(self):
        episodes = int(self.eval_config["episodes"])
        seeds = self.eval_config["seeds"]
        metrics = self._evaluate(episodes, seeds)
        self._log_collect_csv(
            self.total_steps,
            metrics["reward_mean"],
            metrics["reward_std"],
            metrics["elapsed"],
            filename="metrics.csv",
        )
