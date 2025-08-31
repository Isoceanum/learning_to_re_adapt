#!/usr/bin/env python3
"""
Evaluate PPO (Stable-Baselines3) on HalfCheetahCustom-v0.

This script is written with academic research standards:
- Loads a trained PPO model from outputs/
- Runs evaluation episodes deterministically
- Supports rendering for visual inspection
- Reports per-episode and mean reward, aligned with reproducibility norms
"""

from pathlib import Path
import gymnasium
import envs   # ensure HalfCheetahCustom-v0 is registered
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Define repo structure
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "hg_ppo_half_cheetah"
MODEL_PATH = OUTPUT_DIR / "ppo_half_cheetah.zip"


def make_env(render: bool = False):
    """Return a monitored DummyVecEnv for evaluation."""
    def _init():
        env = gymnasium.make("HalfCheetahCustom-v0", render_mode="human" if render else None)
        env = Monitor(env)
        return env
    return DummyVecEnv([_init])


def eval_hf_ppo_half_cheetah(episodes: int = 5, render: bool = False):
    """
    Evaluate a saved PPO agent on HalfCheetahCustom-v0.

    Args:
        episodes (int): Number of evaluation episodes.
        render (bool): If True, render each step visually.

    Returns:
        float: Mean reward across evaluation episodes.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

    env = make_env(render=render)
    model = PPO.load(str(MODEL_PATH))

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
        print(f"Episode {ep+1}/{episodes}: reward = {total_reward:.2f}")

    env.close()
    mean_reward = sum(all_rewards) / len(all_rewards)
    print(f"\n✅ Mean reward over {episodes} episodes: {mean_reward:.2f}")
    return mean_reward


if __name__ == "__main__":
    # By default: run 3 visual episodes for qualitative inspection
    eval_hf_ppo_half_cheetah(episodes=3, render=True)
