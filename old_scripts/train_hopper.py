#!/usr/bin/env python3
"""
Train PPO (Stable-Baselines3) on HopperCustom-v0.

Mirrors the style of scripts/train_half_cheetah.py with minor adjustments for Hopper.
Includes seeding, optional vectorized envs, and SB3 PPO hyperparameters
commonly used to solve Mujoco Hopper reliably.
"""

import random
from pathlib import Path

import gymnasium
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Trigger custom env registration (HopperCustom-v0)
import envs  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(rank: int, render: bool = False, seed: int = 0):
    """Create a single monitored HopperCustom-v0 env with unique seed."""

    def _init():
        env = gymnasium.make(
            "HopperCustom-v0",
            render_mode="human" if render else None,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def train_hopper(
    total_timesteps: int = 5_000_000,
    render: bool = False,
    experiment_name: str = "hg_ppo_hopper",
    seed: int = 0,
    n_envs: int = 1,
):
    """Train PPO on HopperCustom-v0.

    Args:
        total_timesteps: Number of training timesteps.
        render: If True, render during training (usually False for speed).
        experiment_name: Folder under outputs/ to save logs and model.
        seed: Base seed for reproducibility.
        n_envs: Parallel environments (>=2 uses SubprocVecEnv).
    """

    output_dir = REPO_ROOT / "outputs" / experiment_name
    (output_dir / "tb").mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    if n_envs == 1:
        env = DummyVecEnv([make_env(0, render=render, seed=seed)])
    else:
        env = SubprocVecEnv([make_env(i, render=False, seed=seed) for i in range(n_envs)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=str(output_dir / "tb"),
        seed=seed,
        # SB3 baseline-like hyperparameters for Hopper
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    model.learn(total_timesteps=int(total_timesteps), progress_bar=True)
    model.save(str(output_dir / "ppo_hopper"))

    env.close()
    print(f"✅ Model saved to {output_dir}/ppo_hopper.zip")


if __name__ == "__main__":
    train_hopper()

