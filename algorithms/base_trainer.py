import csv
import os
from perturbations.perturbation_factory import resolve_perturbation_env
from utils.seed import seed_env, set_seed
import time
import numpy as np
import envs, gymnasium as gym
import torch

class BaseTrainer:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.env_id = config["env"]
        self.env = None
        self.train_config = config.get("train", {})
        self.eval_config = config.get("eval", {})
        self.eval_interval_steps = int(self.train_config["eval_interval_steps"])
        
        self._global_env_step_counter = 0
        self._steps_since_eval = 0
        
        
        self.device = self._resolve_device()
        print("Using device : ", self.device)
        self.train_seed = self.train_config["seed"]
    
    def _make_train_env(self):
        env = gym.make(self.env_id, exclude_current_positions_from_observation=False)
        env = resolve_perturbation_env(env, self.train_config, self.train_seed)
        env.reset(seed=self.train_seed)
        seed_env(env, self.train_seed)
        return env

    def _make_eval_env(self, seed):
        env = gym.make(self.env_id, exclude_current_positions_from_observation=False)
        env = resolve_perturbation_env(env, self.eval_config, seed)
        env.reset(seed=seed)
        seed_env(env, seed)
        return env
    
    def evaluate_checkpoint(self):
        metrics = self._evaluate(2, [0,1,2,3,4])
        
        metrics_path = os.path.join(self.output_dir, "metrics.csv")
        write_header = not os.path.isfile(metrics_path)
        
        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)
                
    def _evaluate(self, episodes, seeds):
        all_rewards = []
        forward_progresses = []
        episode_lengths = []
        eval_start_time = time.time()
        
        for seed in seeds:
            set_seed(seed)
            eval_env = self._make_eval_env(seed=seed)
            seed_rewards = []
            seed_forward = []
            seed_lengths = []
            
            for episode in range(episodes):
                obs, _ = eval_env.reset()
                com_x_start = None

                done = False
                ep_reward = 0.0
                steps = 0
                last_com_x = None

                while not done:
                    action = self.predict(obs)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    ep_reward += float(reward)
                    steps += 1
                    
                    if com_x_start is None:
                        com_x_start = self._get_forward_position(info)
                    last_com_x = self._get_forward_position(info)
                    
                # Compute forward progress
                fp = last_com_x - com_x_start if (com_x_start is not None and last_com_x is not None) else 0.0
                
                seed_rewards.append(ep_reward)
                seed_forward.append(fp)
                seed_lengths.append(steps)
                all_rewards.append(ep_reward)
                forward_progresses.append(fp)
                episode_lengths.append(steps)
            eval_env.close()
        
        return {
            "time_steps": self._global_env_step_counter,
            "reward_mean": np.mean(all_rewards),
            "reward_std": np.std(all_rewards),
            "forward_progress_mean": np.mean(forward_progresses),
            "forward_progress_std": np.std(forward_progresses),
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
        print(f"- forward_progress_mean: {metrics['forward_progress_mean']:.4f}")
        print(f"- forward_progress_std: {metrics['forward_progress_std']:.4f}")
        print(f"- episode_length_mean: {metrics['episode_length_mean']:.2f}")
        print(f"- elapsed: {elapsed_str}")
         
    def _step_env(self, action):
        self._steps_since_eval += 1
        self._global_env_step_counter += 1
        if self.eval_interval_steps and self.eval_interval_steps > 0 and self._steps_since_eval >= self.eval_interval_steps:
            self._steps_since_eval = 0
            self.evaluate_checkpoint()        
        
        return self.env.step(action)
          
    def _get_forward_position(self, info):
        if "x_position" not in info:
            raise KeyError("Missing x_position in info.")
        return float(info["x_position"])

    def _resolve_device(self):
        device = self.config["device"].lower()
        cuda_available = torch.cuda.is_available()
        
        if device == "auto":
            return torch.device("cuda" if cuda_available else "cpu")
        
        if device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA requested but is not available")
        

        return torch.device(device)
        
    def set_eval_config(self, eval_config):
        # Helper method used by the evaluate_experiment to overwrite eval config
        self.eval_config = eval_config

    def train(self):
       raise NotImplementedError("train() must be implemented in subclass")

    def predict(self, obs):
        raise NotImplementedError("predict() must be implemented in subclass")
    
    def load(self, path):
        raise NotImplementedError("load() must be implemented in subclass")
    
    def save(self):
        raise NotImplementedError("save() must be implemented in subclass")
