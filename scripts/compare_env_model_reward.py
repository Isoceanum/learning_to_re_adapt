"""
Run random rollouts in a Gymnasium env and compare:
  - the env's true reward from `env.step(...)`
  - the model-based reward from `env.unwrapped.get_model_reward_fn()`

Usage:
  python scripts/compare_env_model_reward.py GymPusher-v0
"""

from __future__ import annotations

import sys

import numpy as np

# Ensure custom envs are registered (e.g., GymPusher-v0, HalfCheetahNagabandi-v0)
import envs  # noqa: F401
import gymnasium as gym


# Keep knobs as constants so the only terminal input is the env name.
SEED = 0
NUM_EPISODES = 1000
PRINT_FIRST_STEPS = 0  # per episode
MAX_STEPS = None  # None -> use env's TimeLimit


def _as_torch(x):
    import torch

    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x, dtype=torch.float32)


def _to_float(x) -> float:
    import torch

    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/compare_env_model_reward.py <env_id>")
        return 2

    env_id = sys.argv[1]

    try:
        import torch  # noqa: F401
    except Exception as e:
        print("This script requires torch because get_model_reward_fn() uses torch tensors.")
        print(f"Import error: {e}")
        return 1

    env = gym.make(env_id)
    try:
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "get_model_reward_fn"):
            print(f"{env_id} does not expose `get_model_reward_fn()` on env.unwrapped.")
            return 1

        model_reward_fn = unwrapped.get_model_reward_fn()

        print(f"Env: {env_id}")
        print(f"Obs space: {env.observation_space}")
        print(f"Act space: {env.action_space}")
        if getattr(env, "spec", None) is not None:
            print(f"Max episode steps (spec): {env.spec.max_episode_steps}")
        print()

        all_step_diffs: list[float] = []
        all_env_rewards: list[float] = []
        all_model_rewards: list[float] = []

        for ep in range(NUM_EPISODES):
            obs, _info = env.reset(seed=SEED + ep)

            ep_env_return = 0.0
            ep_model_return = 0.0
            ep_diffs: list[float] = []

            step = 0
            done = False
            while not done:
                if MAX_STEPS is not None and step >= MAX_STEPS:
                    break

                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated) or bool(truncated)

                env_reward = float(reward)
                model_reward = _to_float(
                    model_reward_fn(_as_torch(obs), _as_torch(action), _as_torch(next_obs))
                )
                diff = model_reward - env_reward

                ep_env_return += env_reward
                ep_model_return += model_reward
                ep_diffs.append(diff)

                all_env_rewards.append(env_reward)
                all_model_rewards.append(model_reward)
                all_step_diffs.append(diff)

                if step < PRINT_FIRST_STEPS:
                    extra_parts: list[str] = []
                    if isinstance(info, dict):
                        for k in ("reward_dist", "reward_near", "reward_ctrl"):
                            if k in info:
                                extra_parts.append(f"{k}={float(info[k]):+.6f}")
                    extra = f" ({', '.join(extra_parts)})" if extra_parts else ""
                    print(
                        f"ep={ep:02d} step={step:03d} env={env_reward:+.6f} "
                        f"model={model_reward:+.6f} diff={diff:+.6f}{extra}"
                    )

                obs = next_obs
                step += 1

            ep_diffs_np = np.asarray(ep_diffs, dtype=np.float64)
            ep_abs = np.abs(ep_diffs_np)
            max_abs = float(ep_abs.max()) if ep_abs.size else 0.0
            mean_abs = float(ep_abs.mean()) if ep_abs.size else 0.0
            mean_signed = float(ep_diffs_np.mean()) if ep_diffs_np.size else 0.0

            print(
                f"Episode {ep + 1}/{NUM_EPISODES}: steps={step} "
                f"return_env={ep_env_return:+.6f} return_model={ep_model_return:+.6f} "
                f"return_diff={(ep_model_return - ep_env_return):+.6f} "
                f"(mean|diff|={mean_abs:.6f}, max|diff|={max_abs:.6f}, mean_diff={mean_signed:+.6f})"
            )
            print()

        diffs_np = np.asarray(all_step_diffs, dtype=np.float64)
        abs_np = np.abs(diffs_np)
        env_np = np.asarray(all_env_rewards, dtype=np.float64)
        model_np = np.asarray(all_model_rewards, dtype=np.float64)

        print("Overall:")
        print(f"- total_steps: {diffs_np.size}")
        if diffs_np.size:
            corr = float(np.corrcoef(env_np, model_np)[0, 1]) if diffs_np.size > 1 else float("nan")
            worst_idx = int(abs_np.argmax())
            print(f"- mean|diff|: {float(abs_np.mean()):.6f}")
            print(f"- max|diff|:  {float(abs_np.max()):.6f} (step_idx={worst_idx})")
            print(f"- mean_diff:  {float(diffs_np.mean()):+.6f}")
            print(f"- corr(env, model): {corr:.6f}")

        return 0
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
