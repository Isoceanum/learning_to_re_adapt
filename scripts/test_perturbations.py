"""Quick check that perturbation configs affect training and evaluation envs."""

import argparse
import os
import tempfile

import numpy as np
import yaml

from envs.perturbation_wrapper import PerturbationWrapper


def _build_trainer(config, output_dir):
    algo = str(config.get("algo", "")).lower()
    if algo == "mb_mpc":
        from algorithms.mb_mpc import MBMPCTrainer

        return MBMPCTrainer(config, output_dir)
    if algo == "ppo":
        from algorithms.ppo.ppo_trainer import PPOTrainer

        return PPOTrainer(config, output_dir)
    if algo == "trpo":
        from algorithms.trpo.trpo_trainer import TrpoTrainer

        return TrpoTrainer(config, output_dir)
    raise ValueError(f"Unsupported algo '{algo}' in config")


def _iter_wrappers(env):
    if hasattr(env, "envs"):
        for sub_env in env.envs:
            yield from _iter_wrappers(sub_env)
        return

    current = env
    while True:
        if isinstance(current, PerturbationWrapper):
            yield current
        if not hasattr(current, "env"):
            break
        current = current.env


def _describe_episode(wrapper, episode_idx):
    perturb = wrapper.perturbation
    desc = str(perturb)
    sample = None
    delta = None
    space = getattr(wrapper, "action_space", None)
    if space is not None and hasattr(space, "shape") and space.shape:
        sample = np.ones(space.shape, dtype=np.float32)
        perturbed = perturb.apply_action(sample)
        delta = perturbed - sample
    return {
        "episode": episode_idx,
        "active": getattr(perturb, "active", False),
        "desc": desc,
        "example_delta": None if delta is None else delta.tolist(),
    }


def _inspect_env(env, scope, episodes):
    print(f"\n[{scope.upper()}] Checking perturbations on {type(env).__name__}")
    wrappers = list(_iter_wrappers(env))
    if not wrappers:
        print("  No perturbation wrapper detected.")
        return

    for episode in range(1, episodes + 1):
        env.reset()
        for idx, wrapper in enumerate(wrappers):
            info = _describe_episode(wrapper, episode)
            delta = info["example_delta"]
            print(
                f"  Env#{idx}: episode={info['episode']} active={info['active']} desc={info['desc']}"
            )
            if delta is not None:
                print(f"          example_delta={delta}")


def main():
    parser = argparse.ArgumentParser(description="Inspect perturbation behaviour")
    parser.add_argument("config", type=str, help="Path to YAML config")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to sample per scope")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory (defaults to temp directory)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.output_dir:
        output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = tempfile.mkdtemp(prefix="perturbation_test_")

    trainer = _build_trainer(config, output_dir)

    _inspect_env(trainer.env, "train", args.episodes)

    eval_env = trainer._make_eval_env()
    _inspect_env(eval_env, "eval", args.episodes)

    print(f"\nDone. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
