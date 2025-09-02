#!/usr/bin/env python3
"""
Evaluate PPO (Stable-Baselines3) on HopperCustom-v0.

Matches the style of scripts/eval_half_cheetah.py and reports per-episode
and mean rewards. Supports optional rendering.
"""

from pathlib import Path

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure custom envs are registered
import envs  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "hg_ppo_hopper"
MODEL_PATH = OUTPUT_DIR / "ppo_hopper.zip"


def make_env(render: bool = False):
    """Return a monitored DummyVecEnv for evaluation."""

    def _init():
        env = gymnasium.make("HopperCustom-v0", render_mode="human" if render else None)
        env = Monitor(env)
        return env

    return DummyVecEnv([_init])


def eval_hopper(episodes: int = 5, render: bool = False):
    """
    Evaluate a saved PPO agent on HopperCustom-v0.

    Args:
        episodes: Number of evaluation episodes.
        render: If True, render each step.

    Returns:
        float: Mean episode reward across evaluation episodes.
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


# Backwards-compatible alias (mirrors naming of other eval scripts)
def eval_hf_ppo_hopper(episodes: int = 5, render: bool = False):
    return eval_hopper(episodes=episodes, render=render)


if __name__ == "__main__":
    # Run a few visual episodes by default for quick inspection
    eval_hopper(episodes=3, render=True)
