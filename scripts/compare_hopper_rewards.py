import argparse
from dataclasses import dataclass

import numpy as np
import torch

import envs  # noqa: F401 -- ensures HopperCustom-v0 is registered
import gymnasium as gym


@dataclass
class RewardCheckResult:
    max_abs_diff: float
    mismatches: list[tuple[int, float, float, float]]


def compare_rewards(num_steps: int, seed: int) -> RewardCheckResult:
    env = gym.make("HopperCustom-v0", exclude_current_positions_from_observation=False)
    base_env = env.unwrapped
    reward_fn = base_env.get_model_reward_fn()
    rng = np.random.default_rng(seed)

    state, _ = env.reset(seed=seed)

    mismatches: list[tuple[int, float, float, float]] = []
    max_abs_diff = 0.0

    for step in range(num_steps):
        action = rng.uniform(
            low=env.action_space.low,
            high=env.action_space.high,
        ).astype(np.float32)

        next_state, env_reward, terminated, truncated, _ = env.step(action)

        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        action_t = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)

        model_reward = reward_fn(state_t, action_t, next_state_t).item()
        diff = abs(env_reward - model_reward)

        if diff > max_abs_diff:
            max_abs_diff = diff

        if diff > 1e-5:
            mismatches.append((step, env_reward, model_reward, diff))

        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    env.close()
    return RewardCheckResult(max_abs_diff=max_abs_diff, mismatches=mismatches)


def main():
    parser = argparse.ArgumentParser(
        description="Verify Hopper env reward matches MB-MPC reward function.",
    )
    parser.add_argument("--steps", type=int, default=100, help="Transitions to sample.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()

    result = compare_rewards(num_steps=args.steps, seed=args.seed)

    print(f"Checked {args.steps} transitions.")
    print(f"Max absolute difference: {result.max_abs_diff:.8f}")

    if result.mismatches:
        print("Found mismatches:")
        for step, env_reward, model_reward, diff in result.mismatches:
            print(
                f"  step={step:04d} env_reward={env_reward:.8f} "
                f"model_reward={model_reward:.8f} diff={diff:.8f}"
            )
        raise SystemExit(1)

    print("Rewards match within tolerance.")


if __name__ == "__main__":
    main()
