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
        """Create the training environment."""
        import envs  # Ensure custom envs are registered
        import gymnasium
        from stable_baselines3.common.env_util import make_vec_env

        env_id = self.config.get("env")
        n_envs = int(self.train_config.get("n_envs", self.config.get("n_envs", 1)))

        if n_envs > 1:
            return make_vec_env(env_id, n_envs=n_envs)

        return gymnasium.make(env_id)

    def _make_eval_env(self):
        """Always create a single env for evaluation."""
        import envs  # Ensure custom envs are registered
        import gymnasium

        env_id = self.config.get("env")
        return gymnasium.make(env_id)

    def train(self):
        raise NotImplementedError("train() must be implemented in subclass")

    def evaluate(self):
        """Generic evaluation over seeds × episodes and a single CSV output.

        Subclasses must implement `_predict(self, obs, deterministic)`.
        """
        eval_t0 = time.time()
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
        # Use a standalone, single environment for evaluation to keep the
        # step/reset API simple even if training used vectorized envs.
        eval_env = self._make_eval_env()

        for seed in seeds:
            for ep in range(episodes):
                # Reset with optional seed
                if seed is None:
                    obs, _ = eval_env.reset()
                else:
                    obs, _ = eval_env.reset(seed=int(seed))

                # Nagabandi parity: track torso COM x from observation (last 3 dims are COM)
                try:
                    com_x_start = float(obs[-3])
                except Exception:
                    com_x_start = None

                done = False
                total_reward = 0.0
                discounted_return = 0.0
                discount = 1.0
                steps = 0
                last_com_x = None
                policy_time_ep = 0.0
                env_time_ep = 0.0
                ep_start_time = time.time()

                while not done:
                    t0 = time.time()
                    action = self._predict(obs, deterministic)
                    policy_time_ep += time.time() - t0

                    t1 = time.time()
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    env_time_ep += time.time() - t1

                    r = float(reward)
                    total_reward += r
                    discounted_return += discount * r
                    discount *= gamma
                    steps += 1
                    done = terminated or truncated

                    # Update COM x from observation exclusively (Nagabandi semantics)
                    try:
                        last_com_x = float(obs[-3])
                    except Exception:
                        pass

                # Forward progress strictly from COM (no fallback)
                fp = (last_com_x - com_x_start) if (com_x_start is not None and last_com_x is not None) else 0.0
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
        fp_std = pstdev(forward_progresses) if len(forward_progresses) > 1 else 0.0

        elapsed = time.time() - eval_t0
        elapsed_str = f"{int(elapsed)//3600:02d}:{(int(elapsed)%3600)//60:02d}:{int(elapsed)%60:02d}"
        print("\n✅ Evaluation summary:")
        print(f"- reward_mean: {mean_reward:.4f}")
        print(f"- reward_std: {std_reward:.4f}")
        print(f"- forward_progress_mean: {fp_mean:.4f}")
        print(f"- forward_progress_std: {fp_std:.4f}")
        print(f"- elapsed: {elapsed_str}")

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

        csv_path = os.path.join(self.output_dir, "results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"💾 Saved evaluation results to {csv_path}")

        try:
            eval_env.close()
        except Exception:
            pass

        return mean_reward

    def save(self):
        path = os.path.join(self.output_dir, "model")
        self.model.save(path)

    def load(self, path):
        raise NotImplementedError("load() must be implemented in subclass")

    # Hook to be implemented by subclasses for action selection
    def _predict(self, obs, deterministic: bool):
        raise NotImplementedError("_predict() must be implemented in subclass")
