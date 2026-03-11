import argparse
import numpy as np
import gymnasium as gym

import envs  # registers custom envs
from perturbations.cripple_perturbation import CripplePerturbation


def _parse_args():
    parser = argparse.ArgumentParser(description="Sanity-check FaithfulAnt + cripple perturbation.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="FaithfulAnt-v0")
    return parser.parse_args()


def main():
    args = _parse_args()

    env = gym.make(args.env)
    perturbation_config = {
        "type": "cripple",
        "probability": 1.0,
        "candidate_action_indices": [[0, 1], [2, 3], [4, 5], [6, 7]],
    }
    env = CripplePerturbation(env, perturbation_config, seed=args.seed)

    print(f"Env: {args.env}")
    print(f"Episodes: {args.episodes} | Steps: {args.steps}")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        base_env = getattr(env, "unwrapped", env)
        disabled_leg = getattr(base_env, "_disabled_leg", None)
        sampled = env.sampled_indices
        active = env.is_active()
        print(f"\nEpisode {ep}: active={active} sampled_indices={sampled} disabled_leg={disabled_leg}")

        for t in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
