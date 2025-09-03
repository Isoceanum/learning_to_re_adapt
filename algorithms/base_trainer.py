# algorithms/base_trainer.py
import os
import csv
import time
from statistics import pstdev


class BaseTrainer:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.train_config = config.get("train", {})
        self.eval_config = config.get("eval", {})
        self.env = None

    def _make_env(self):
        import envs 
        import gymnasium
        env_id = self.config.get("env")
        return gymnasium.make(env_id)

    def train(self):
        raise NotImplementedError("train() must be implemented in subclass")

    def evaluate(self):
        """Generic evaluation over seeds × episodes and a single CSV output.

        Subclasses must implement `_predict(self, obs, deterministic)`.
        """
        # Settings
        episodes = int(self.eval_config.get("episodes", 10))
        seeds = self.eval_config.get("seeds", [None])
        if not isinstance(seeds, list) or len(seeds) == 0:
            seeds = [None]
        deterministic = bool(self.eval_config.get("deterministic", True))
        gamma = float(self.train_config.get("gamma", 0.99))

        # Minimal run metadata to keep CSV tidy
        meta = {
            "episodes_per_seed": episodes,
            "deterministic": deterministic,
        }

        # Storage
        all_rewards = []
        forward_progresses = []
        rows = []

        total_runs = len(seeds) * episodes
        print(f"🎯 Evaluating over {episodes} episode(s) per seed × {len(seeds)} seed(s) = {total_runs} episodes")

        # Loop
        for seed in seeds:
            for ep in range(episodes):
                # Reset with optional seed
                if seed is None:
                    obs, _ = self.env.reset()
                else:
                    obs, _ = self.env.reset(seed=int(seed))

                # Starting x position (MuJoCo) with safe default
                try:
                    x_start = float(self.env.unwrapped.data.qpos[0])
                except Exception:
                    x_start = 0.0

                done = False
                total_reward = 0.0
                discounted_return = 0.0
                discount = 1.0
                steps = 0
                last_x_pos = None
                policy_time_ep = 0.0
                env_time_ep = 0.0
                ep_start_time = time.time()

                while not done:
                    t0 = time.time()
                    action = self._predict(obs, deterministic)
                    policy_time_ep += time.time() - t0

                    t1 = time.time()
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    env_time_ep += time.time() - t1

                    r = float(reward)
                    total_reward += r
                    discounted_return += discount * r
                    discount *= gamma
                    steps += 1
                    done = terminated or truncated

                    if isinstance(info, dict) and "x_position" in info:
                        try:
                            last_x_pos = float(info["x_position"])
                        except Exception:
                            pass

                # Determine end x; fallback to current state
                if last_x_pos is not None:
                    x_end = last_x_pos
                else:
                    try:
                        x_end = float(self.env.unwrapped.data.qpos[0])
                    except Exception:
                        x_end = x_start

                fp = x_end - x_start
                all_rewards.append(total_reward)
                forward_progresses.append(fp)

                rows.append({
                    "episode": ep,
                    "seed": "" if seed is None else int(seed),
                    "reward": total_reward,
                    "discounted_return": discounted_return,
                    "forward_progress": fp,
                    "steps": steps,
                    "policy_exec_time_s": policy_time_ep,
                    "env_exec_time_s": env_time_ep,
                    "episode_time_s": time.time() - ep_start_time,
                    **meta,
                })

        # Summary
        if not all_rewards:
            print("No evaluation episodes were run.")
            return 0.0

        mean_reward = sum(all_rewards) / len(all_rewards)
        std_reward = pstdev(all_rewards) if len(all_rewards) > 1 else 0.0
        fp_mean = (sum(forward_progresses) / len(forward_progresses)) if forward_progresses else 0.0

        print("\n✅ Evaluation summary:")
        print(f"- reward_mean: {mean_reward:.4f}")
        print(f"- reward_std: {std_reward:.4f}")
        print(f"- forward_progress_mean: {fp_mean:.4f}")

        # Single CSV output with fixed schema
        fieldnames = [
            "episode",
            "seed",
            "reward",
            "discounted_return",
            "forward_progress",
            "steps",
            "policy_exec_time_s",
            "env_exec_time_s",
            "episode_time_s",
            "episodes_per_seed",
            "deterministic",
        ]

        csv_path = os.path.join(self.output_dir, "eval_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"💾 Saved evaluation results to {csv_path}")

        return mean_reward

    def save(self):
        path = os.path.join(self.output_dir, "model")
        self.model.save(path)

    def load(self, path):
        raise NotImplementedError("load() must be implemented in subclass")

    # Hook to be implemented by subclasses for action selection
    def _predict(self, obs, deterministic: bool):
        raise NotImplementedError("_predict() must be implemented in subclass")
