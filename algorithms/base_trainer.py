import os


class BaseTrainer:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.env_id = config["env"]
        self.env = None
        self.train_config = config.get("train", {})
        self.eval_config = config.get("eval", {})

        self.train_seed = self.train_config["seed"]
    
    def _make_train_env(self):
        import envs, gymnasium as gym
        env = gym.make(self.env_id)
        env.reset(seed=self.train_seed)
        return env

    def _make_eval_env(self, seed):
        import envs, gymnasium as gym
        env = gym.make(self.env_id)
        env.reset(seed=seed)
        return env
        
    def evaluate(self):
        import time
        import numpy as np

        episodes = int(self.eval_config["episodes"])
        seeds = self.eval_config["seeds"]

        all_rewards = []
        forward_progresses = []
        total_runs = len(seeds) * episodes

        print(f"Evaluating {episodes} episode(s) × {len(seeds)} seed(s) = {total_runs} total runs")

        eval_start_time = time.time()

        for seed in seeds:
            for episode in range(episodes):
                eval_seed = int(seed) * 1000 + episode
                eval_env = self._make_eval_env(seed=eval_seed)
                obs, _ = eval_env.reset(seed=eval_seed)

                # Track forward progress (last 3 obs dims as COM x, Nagabandi convention)
                try:
                    com_x_start = float(obs[-3])
                except Exception:
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

                    try:
                        last_com_x = float(obs[-3])
                    except Exception:
                        pass

                # Compute forward progress
                if com_x_start is not None and last_com_x is not None:
                    fp = last_com_x - com_x_start
                else:
                    fp = 0.0

                all_rewards.append(ep_reward)
                forward_progresses.append(fp)
                eval_env.close()

        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        fp_mean = np.mean(forward_progresses)
        fp_std = np.std(forward_progresses)
        elapsed = time.time() - eval_start_time
        elapsed_str = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"

        print("\nEvaluation summary:")
        print(f"- reward_mean: {mean_reward:.4f}")
        print(f"- reward_std: {std_reward:.4f}")
        print(f"- forward_progress_mean: {fp_mean:.4f}")
        print(f"- forward_progress_std: {fp_std:.4f}")
        print(f"- elapsed: {elapsed_str}")

    def train(self):
       raise NotImplementedError("train() must be implemented in subclass")

    def predict(self, obs):
        raise NotImplementedError("predict() must be implemented in subclass")
    
    def load(self, path):
        raise NotImplementedError("load() must be implemented in subclass")
    
    def save(self):
        raise NotImplementedError("save() must be implemented in subclass")