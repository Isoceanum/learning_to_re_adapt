#!/usr/bin/env python3
"""
Evaluate PPO (Stable-Baselines3) on Ant.

This is a baseline reference script:
- Loads PPO model from outputs/hf_ppo_ant/ppo_ant.zip
- Runs evaluation episodes with raw (non-normalized) environment
- Can render Ant during rollout if specified
"""

from pathlib import Path
from stable_baselines3 import PPO
from envs.ant_env import AntEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "hf_ppo_ant"


def make_env(render: bool = False):
    def _init():
        env = AntEnv(render_mode="human" if render else None)
        env = Monitor(env)
        return env
    return DummyVecEnv([_init])


def eval_hf_ppo_ant(episodes: int = 5, render: bool = False):
    """Evaluate a saved PPO agent on Ant."""
    model_path = OUTPUT_DIR / "ppo_ant.zip"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    env = make_env(render=render)
    model = PPO.load(str(model_path))

    all_rewards = []
    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            reward = float(rewards[0])
            done = bool(dones[0])

            total_reward += reward

            if render:
                env.render()

        all_rewards.append(total_reward)
        print(f"Episode {ep+1}: reward={total_reward:.2f}")

    env.close()
    mean_reward = sum(all_rewards) / len(all_rewards)
    print(f"✅ Mean reward over {episodes} episodes: {mean_reward:.2f}")
    return mean_reward


if __name__ == "__main__":
    eval_hf_ppo_ant(episodes=3, render=True)
