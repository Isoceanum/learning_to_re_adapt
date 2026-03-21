import csv
import os
from perturbations.perturbation_factory import resolve_perturbation_env, wrap_perturbation
from utils.seed import seed_env, set_seed
import time
import numpy as np
import envs, gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import torch
import functools


def build_env_from_config(environment_config, perturbation_config, seed):
    env_id = environment_config["id"]
    env_kwargs = {}
    env_kwargs["exclude_current_positions_from_observation"] = environment_config["exclude_current_positions_from_observation"]

    env = gym.make(env_id, **env_kwargs)
    env = wrap_perturbation(env, perturbation_config, seed)
    env.reset(seed=seed)
    seed_env(env, seed)
    return env


class BaseTrainer:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        
        self.environment_config = config["environment"]
        self.train_config = config["train"]
        self.eval_config = config["eval"]
        
        self.env_id = self.environment_config["id"]
                
        self.device = self._resolve_device()
        print("Using device : ", self.device)
        
        self.train_seed = self.train_config["seed"]
        self.num_parallel_envs = self.environment_config.get("num_parallel_envs", 1)
        
        self.env = self._make_env(self.environment_config, None, self.train_seed)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        
    def _make_env(self, environment_config, perturbation_config, seed):
        return build_env_from_config(environment_config, perturbation_config, seed)
    
    def _evaluate(self, episodes, seeds):
        all_rewards = []
        episode_lengths = []
        eval_start_time = time.time()
        eval_perturbation_config = self.eval_config.get("perturbation", {})
        
        max_episode_length = self.environment_config["max_episode_length"]
        self.environment_config
        
        for seed in seeds:
            set_seed(seed)
            eval_env = self._make_env(self.environment_config, eval_perturbation_config, seed)
            seed_rewards = []
            seed_lengths = []
            
            for episode in range(episodes):
                if hasattr(self, "_reset_eval_adaptation"):
                    self._reset_eval_adaptation()
                obs, _ = eval_env.reset()
                done = False
                ep_reward = 0.0
                steps = 0

                while not done:
                    action = self.predict(obs)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    ep_reward += float(reward)
                    steps += 1

                    # Respect an explicit max path length for evaluation if set.
                    if steps >= max_episode_length:
                        done = True
                        truncated = True

                seed_rewards.append(ep_reward)
                seed_lengths.append(steps)
                all_rewards.append(ep_reward)
                episode_lengths.append(steps)
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
         

    def collect_steps(self, iteration_index, steps_target, max_path_length, add_fn):
        setup_start_time = time.time()
        parallel_env_seeds = [self.train_seed + i for i in range(self.num_parallel_envs)]
        
        train_perturbation_config = self.train_config.get("perturbation", {})
        
        parallel_env_factories = []
        for seed in parallel_env_seeds:
            parallel_env_factories.append(functools.partial(build_env_from_config, self.environment_config, train_perturbation_config, seed))

        parallel_env = AsyncVectorEnv(parallel_env_factories)
        obs, _ = parallel_env.reset(seed=parallel_env_seeds)
        setup_time = time.time() - setup_start_time

        # Per-env quota for a simpler collection loop (allows small overshoot).
        steps_per_env = int(np.ceil(steps_target / self.num_parallel_envs))
        steps_collected_per_env = [0 for _ in range(self.num_parallel_envs)]

        # Global collection counters and timing.
        steps_collected_this_iteration = 0
        log_collect_start_time = time.time()
        log_episodes = 0
        action_time = 0.0
        step_time = 0.0
        reset_time = 0.0
        adapt_time = 0.0
        plan_time = 0.0
        adapted_envs = 0
        action_envs = 0

        # Episode-level metrics to aggregate once collection ends.
        log_episode_returns = []
        log_episode_lengths = []

        # Per-env episode buffers (store full trajectories until an episode ends).
        episode_obs = [[] for _ in range(self.num_parallel_envs)]
        episode_act = [[] for _ in range(self.num_parallel_envs)]
        episode_next_obs = [[] for _ in range(self.num_parallel_envs)]
        # Per-env episode metrics (accumulated while the episode is active).
        episode_returns = [0.0 for _ in range(self.num_parallel_envs)]
        episode_steps = [0 for _ in range(self.num_parallel_envs)]

        # Mask of envs that are still collecting toward their per-env quota.
        active_env_mask = np.ones((self.num_parallel_envs,), dtype=np.bool_)

        # Main collection loop: each env collects until it hits its quota,
        # then finishes the current episode and stops contributing.
        while np.any(active_env_mask):

            # Build action list for all envs.
            # We keep stepping all envs, but only active envs contribute
            # transitions and counts.
            active_indices = [i for i in range(self.num_parallel_envs) if active_env_mask[i]]
            actions = [self.env.action_space.sample() for _ in range(self.num_parallel_envs)]
            if active_indices:
                obs_batch = np.stack([obs[i] for i in active_indices], axis=0)
                action_start_time = time.time()
                batch_actions = self._batch_predict(obs_batch, active_indices, iteration_index)
                action_time += time.time() - action_start_time
                if hasattr(self, "_last_adapt_time"):
                    adapt_time += self._last_adapt_time
                if hasattr(self, "_last_plan_time"):
                    plan_time += self._last_plan_time
                if hasattr(self, "_last_adapted_count"):
                    adapted_envs += self._last_adapted_count
                if hasattr(self, "_last_action_envs"):
                    action_envs += self._last_action_envs
                if torch.is_tensor(batch_actions):
                    batch_actions = batch_actions.detach().cpu().numpy()
                for j, i in enumerate(active_indices):
                    actions[i] = batch_actions[j]

            # Step all envs simultaneously.
            step_start_time = time.time()
            next_obs, rewards, terminated, truncated, _ = parallel_env.step(actions)
            step_time += time.time() - step_start_time
            terminated = np.asarray(terminated, dtype=np.bool_)
            truncated = np.asarray(truncated, dtype=np.bool_)

            # Update global counters only for envs still collecting.
            active_steps = int(np.sum(active_env_mask))
            steps_collected_this_iteration += active_steps

            # Decide which envs finished an episode this step.
            done_mask = np.zeros((self.num_parallel_envs,), dtype=np.bool_)
            for i in range(self.num_parallel_envs):
                if active_env_mask[i]:
                    # Store transition in the per-env episode buffer.
                    episode_obs[i].append(obs[i])
                    episode_act[i].append(actions[i])
                    episode_next_obs[i].append(next_obs[i])

                    # Accumulate per-episode metrics.
                    episode_returns[i] += float(rewards[i])

                    steps_collected_per_env[i] += 1
                    episode_steps[i] += 1

                    if hasattr(self, "support_window_queues") and self.support_window_queues is not None:
                        self.support_window_queues[i].append((obs[i], actions[i], next_obs[i]))

                # Episode ends on max_path_length or termination/truncation.
                if active_env_mask[i] and episode_steps[i] >= max_path_length:
                    done_mask[i] = True
                elif terminated[i] or truncated[i]:
                    done_mask[i] = True

            for i in range(self.num_parallel_envs):
                if not done_mask[i]:
                    continue

                # If this env was contributing, commit the completed trajectory.
                if active_env_mask[i]:
                    add_fn(episode_obs[i], episode_act[i], episode_next_obs[i])
                    log_episodes += 1
                    log_episode_returns.append(float(episode_returns[i]))
                    log_episode_lengths.append(int(episode_steps[i]))

                # Clear per-env episode buffers for the next episode.
                episode_obs[i] = []
                episode_act[i] = []
                episode_next_obs[i] = []
                episode_returns[i] = 0.0
                episode_steps[i] = 0

                if hasattr(self, "support_window_queues") and self.support_window_queues is not None:
                    self._reset_support_window_queues([i])

                # Stop collecting from this env once it hits its quota.
                if steps_collected_per_env[i] >= steps_per_env:
                    active_env_mask[i] = False

            # Reset only the envs that finished, keep others running.
            if np.any(done_mask):
                reset_start_time = time.time()
                obs, _ = parallel_env.reset(seed=parallel_env_seeds, options={"reset_mask": done_mask})
                reset_time += time.time() - reset_start_time
            else:
                obs = next_obs

        parallel_env.close()

        # Aggregate per-episode metrics into a single summary for logging.
        reward_mean = float(np.mean(log_episode_returns)) if log_episode_returns else 0.0
        reward_std = float(np.std(log_episode_returns)) if log_episode_returns else 0.0
        avg_episode_length = float(np.mean(log_episode_lengths)) if log_episode_lengths else 0.0

        log_collect_time = time.time() - log_collect_start_time
        print(
            f"collect: envs={self.num_parallel_envs} steps_target={steps_target} steps_per_env={steps_per_env}"
        )
        print(
            f"collect: steps={steps_collected_this_iteration} "
            f"episodes={log_episodes} "
            f"avg_rew={reward_mean:.3f} "
            f"rew_std={reward_std:.3f} "
            f"avg_ep_len={avg_episode_length:.1f} "
            f"time={log_collect_time:.1f}s"
        )
        print(
            f"collect_timing: setup={setup_time:.2f}s action={action_time:.2f}s "
            f"adapt={adapt_time:.2f}s plan={plan_time:.2f}s "
            f"step={step_time:.2f}s reset={reset_time:.2f}s total={log_collect_time:.2f}s"
        )
        if action_envs > 0:
            adapt_pct = 100.0 * adapted_envs / action_envs
            print(f"collect_adapt: used={adapted_envs}/{action_envs} ({adapt_pct:.1f}%)")
          
    def _resolve_device(self):
        device = self.config["device"].lower()
        cuda_available = torch.cuda.is_available()
        
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
    
    def _reset_support_window_queues(self, env_indices):
        raise NotImplementedError

    def _batch_predict(self, obs_batch, env_indices, iteration_index):
        raise NotImplementedError
