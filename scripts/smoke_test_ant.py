#!/usr/bin/env python3
"""
Minimal smoke test for the local Ant env at `envs/ant_env.py`.

Usage:
  python scripts/smoke_test_ant.py [--steps 200] [--seed 42] [--render]

This script:
- Loads `envs/ant_env.py` directly from file (no sys.path tweaks).
- Instantiates `AntEnv` with the bundled XML at `envs/assets/ant.xml`.
- Runs a short random rollout and prints simple stats.
"""

from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Optional


def load_local_ant_class():
    repo_root = Path(__file__).resolve().parents[1]
    ant_path = repo_root / "envs" / "ant_env.py"
    spec = spec_from_file_location("local_ant", ant_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {ant_path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, "AntEnv"):
        raise RuntimeError("AntEnv not found in envs/ant_env.py")
    return mod.AntEnv, repo_root


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ant smoke test (local env)")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    args = parser.parse_args(argv)

    # Enforce gymnasium 1.2.0 and avoid gym
    import gymnasium as gym
    version = getattr(gym, "__version__", "")
    if version != "1.2.0":
        raise RuntimeError(f"gymnasium==1.2.0 required, found {version}")

    AntEnv, repo_root = load_local_ant_class()
    xml_path = repo_root / "envs" / "assets" / "ant.xml"

    # Prefer passing render_mode when requested; keep it simple otherwise.
    if args.render:
        try:
            env = AntEnv(xml_file=str(xml_path), render_mode="human")
        except TypeError:
            env = AntEnv(xml_file=str(xml_path))
    else:
        env = AntEnv(xml_file=str(xml_path))

    # Reset with seed if supported.
    try:
        obs, info = env.reset(seed=args.seed)
    except TypeError:
        out = env.reset()
        obs, info = (out if isinstance(out, tuple) else (out, {}))

    total_reward = 0.0
    last_x_vel = None
    last_forward_reward = None

    render_mode = getattr(env, "render_mode", None)
    call_render_each_step = bool(args.render and render_mode != "human")

    steps_executed = 0
    for _ in range(int(args.steps)):
        action = env.action_space.sample()
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out

        total_reward += float(reward)
        if isinstance(info, dict):
            last_x_vel = info.get("x_velocity", last_x_vel)
            last_forward_reward = info.get(
                "forward_reward", info.get("reward_forward", last_forward_reward)
            )

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
        f"Ant smoke OK — steps={steps_executed} total_reward={total_reward:.4f} "
        f"mean={mean_reward:.4f} last_x_vel={x_vel_str} last_forward_reward={fwd_str}"
    )

    try:
        env.close()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
