#!/usr/bin/env python3
"""
Train PPO (Stable-Baselines3) on Ant.

This is a baseline reference script:
- Matches SB3 benchmark hyperparameters for Ant.
- Uses Monitor to log episode rewards and lengths.
- Saves model and logs under `outputs/{experiment_name}/`.
"""

from pathlib import Path
from stable_baselines3 import PPO
from envs.ant_env import AntEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_env(render: bool = False):
    def _init():
        env = AntEnv(render_mode="human" if render else None)
        env = Monitor(env)  # log rewards and lengths
        return env
    return DummyVecEnv([_init])


def train_hf_ppo_ant(
    total_timesteps: int = 5_000_000,
    render: bool = False,
    experiment_name: str = "hf_ppo_ant",
):
    output_dir = REPO_ROOT / "outputs" / experiment_name
    (output_dir / "tb").mkdir(parents=True, exist_ok=True)

    env = make_env(render=render)   
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=str(output_dir / "tb"),
    )

    model.learn(total_timesteps=int(total_timesteps), progress_bar=True)
    model.save(str(output_dir / "ppo_ant"))

    env.close()
    print(f"✅ Model saved to {output_dir}/ppo_ant.zip")


if __name__ == "__main__":
    train_hf_ppo_ant()
