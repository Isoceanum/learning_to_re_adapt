#!/usr/bin/env python3
"""
Minimal smoke test for the local HalfCheetah env at `envs/half_cheetah_env.py`.

Usage:
  python scripts/smoke_test_half_cheetah.py [--steps 200] [--seed 42] [--render]

This script:
- Loads `envs/half_cheetah_env.py` directly from file (no sys.path tweaks).
- Instantiates `HalfCheetahEnv` using the bundled XML at `envs/assets/half_cheetah.xml`.
- Runs a short random rollout and prints simple stats.

It also exposes a `smoke_test_half_cheetah` function you can call from code.
"""

from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Optional


def load_local_half_cheetah_class():
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / "envs" / "half_cheetah_env.py"
    spec = spec_from_file_location("local_half_cheetah", env_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {env_path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, "HalfCheetahEnv"):
        raise RuntimeError("HalfCheetahEnv not found in envs/half_cheetah_env.py")
    return mod.HalfCheetahEnv


def smoke_test_half_cheetah(steps: int = 200, seed: int = 42, render: bool = False) -> dict:
    """Run a simple random-action rollout in HalfCheetah.

    Returns a small dict of summary stats.
    """
    import gymnasium as gym

    version = getattr(gym, "__version__", "")
    if version != "1.2.0":
        raise RuntimeError(f"gymnasium==1.2.0 required, found {version}")

    HalfCheetahEnv = load_local_half_cheetah_class()

    # HalfCheetahEnv uses internal XML_PATH; render_mode is forwarded via **kwargs to MujocoEnv
    if render:
        try:
            env = HalfCheetahEnv(render_mode="human")
        except TypeError:
            env = HalfCheetahEnv()
    else:
        env = HalfCheetahEnv()

    # Reset with seed if supported
    try:
        obs, info = env.reset(seed=seed)
    except TypeError:
        out = env.reset()
        obs, info = (out if isinstance(out, tuple) else (out, {}))

    total_reward = 0.0
    last_x_vel = None
    last_forward_reward = None

    render_mode = getattr(env, "render_mode", None)
    call_render_each_step = bool(render and render_mode != "human")

    steps_executed = 0
    for _ in range(int(steps)):
        action = env.action_space.sample()
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out

        total_reward += float(reward)
        if isinstance(info, dict):
            # half-cheetah env returns 'x_velocity' and 'reward_run'
            last_x_vel = info.get("x_velocity", last_x_vel)
            last_forward_reward = info.get("reward_run", last_forward_reward)

        if call_render_each_step:
            try:
                env.render()
            except Exception:
                pass

        steps_executed += 1
        if done:
            break

    mean_reward = total_reward / max(1, steps_executed)
    x_vel_str = "None" if last_x_vel is None else f"{last_x_vel:.4f}"
    fwd_str = (
        "None" if last_forward_reward is None else f"{last_forward_reward:.4f}"
    )
    print(
        f"HalfCheetah smoke OK — steps={steps_executed} total_reward={total_reward:.4f} "
        f"mean={mean_reward:.4f} last_x_vel={x_vel_str} last_forward_reward={fwd_str}"
    )

    try:
        env.close()
    except Exception:
        pass

    return {
        "steps": steps_executed,
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "last_x_velocity": last_x_vel,
        "last_forward_reward": last_forward_reward,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HalfCheetah smoke test (local env)")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    args = parser.parse_args(argv)

    smoke_test_half_cheetah(steps=args.steps, seed=args.seed, render=args.render)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

