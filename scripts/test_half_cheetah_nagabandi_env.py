#!/usr/bin/env python
import numpy as np
import gymnasium as gym

# Registers custom env ids, including HalfCheetahNagabandi-v0
import envs  # noqa: F401

def main():
    env_id = "HalfCheetahNagabandi-v0"
    env = gym.make(env_id)
    try:
        obs, info = env.reset(seed=0)
        print(f"Reset ok: obs shape={np.shape(obs)}, info keys={list(info.keys())}")

        action = env.action_space.sample()
        print(
            f"Sampled action shape={action.shape}, "
            f"bounds=({env.action_space.low.min():.2f}, {env.action_space.high.max():.2f})"
        )

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        elif len(step_out) == 4:
            obs, reward, done, info = step_out
            terminated, truncated = done, False
        else:
            raise RuntimeError(f"Unexpected step() output length: {len(step_out)}")

        dt = getattr(env, "dt", getattr(env, "model", None) and env.model.opt.timestep)
        print(
            f"Step ok: reward={reward:.4f}, done={done}, "
            f"terminated={terminated}, truncated={truncated}, obs shape={np.shape(obs)}, dt={dt}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
