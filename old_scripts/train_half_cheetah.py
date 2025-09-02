#!/usr/bin/env python3
import os
from pathlib import Path
import random
import numpy as np
import torch
import gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# trigger env registration
import envs  

REPO_ROOT = Path(__file__).resolve().parents[1]

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_env(rank: int, render: bool = False, seed: int = 0, ctrl_cost_weight: float = 0.5):
    """Utility to create a single env instance with seeding + monitor."""
    def _init():
        env = gymnasium.make(
            "HalfCheetahCustom-v0",
            render_mode="human" if render else None,
            ctrl_cost_weight=ctrl_cost_weight
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)   # each env gets a different seed
        return env
    return _init

def train_half_cheetah(
    total_timesteps: int = 5_000_000,
    render: bool = False,
    experiment_name: str = "hg_ppo_half_cheetah",
    seed: int = 0,
    n_envs: int = 1
):
    """Train PPO on HalfCheetah with configurable number of parallel environments."""
    output_dir = REPO_ROOT / "outputs" / experiment_name
    (output_dir / "tb").mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    if n_envs == 1:
        # single-threaded
        env = DummyVecEnv([make_env(0, render=render, seed=seed)])
    else:
        # parallelized
        env = SubprocVecEnv([make_env(i, render=False, seed=seed) for i in range(n_envs)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=str(output_dir / "tb"),
        seed=seed,
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
    model.save(str(output_dir / "ppo_half_cheetah"))

    env.close()
    print(f"✅ Model saved to {output_dir}/ppo_half_cheetah.zip")

if __name__ == "__main__":
    train_half_cheetah()
