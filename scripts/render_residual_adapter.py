import argparse
import os
import yaml

# Ensure custom envs are registered
import envs  # noqa: F401
import gymnasium as gym

from perturbations.perturbation_factory import resolve_perturbation_env
from scripts.run_experiment import _build_trainer

RENDER_EPISODES = 3


def _resolve_dynamics_model_path(config, run_dir):
    train_cfg = config.get("train", {})
    orig_path = train_cfg.get("dynamics_model_path")
    if not orig_path:
        return
    if os.path.isdir(orig_path):
        return

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    candidates = []

    if "outputs" in orig_path:
        suffix = orig_path[orig_path.index("outputs") :]
        candidates.append(os.path.join(repo_root, suffix))

    if not os.path.isabs(orig_path):
        candidates.append(os.path.abspath(os.path.join(run_dir, orig_path)))
        candidates.append(os.path.abspath(os.path.join(repo_root, orig_path)))

    for cand in candidates:
        if os.path.isdir(cand):
            train_cfg["dynamics_model_path"] = cand
            print(f"Resolved dynamics_model_path -> {cand}")
            return

    print(f"Warning: dynamics_model_path not found locally: {orig_path}")


def main():
    parser = argparse.ArgumentParser(description="Render a residual adapter run.")
    parser.add_argument("run_dir", type=str, help="Run directory containing config.yaml and residual_adapter.pt")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    algo = str(config.get("algo", "")).lower()
    if algo != "residual_adapter":
        print(f"Warning: algo is '{config.get('algo')}', expected 'residual_adapter'")

    _resolve_dynamics_model_path(config, run_dir)
    dynamics_model_path = config.get("train", {}).get("dynamics_model_path")
    if dynamics_model_path and not os.path.isdir(dynamics_model_path):
        raise FileNotFoundError(f"dynamics_model_path not found: {dynamics_model_path}")

    trainer = _build_trainer(config, run_dir)
    env_id = config.get("env")
    trainer.env = gym.make(env_id, render_mode="human", exclude_current_positions_from_observation=False)
    trainer.env = resolve_perturbation_env(trainer.env, trainer.eval_config, 0)

    trainer.load(run_dir)

    try:
        for ep in range(RENDER_EPISODES):
            obs, _ = trainer.env.reset()
            done = False
            ep_return = 0.0
            while not done:
                action = trainer.predict(obs)
                obs, reward, terminated, truncated, _ = trainer.env.step(action)
                done = terminated or truncated
                ep_return += float(reward)
            print(f"Episode {ep + 1}/{RENDER_EPISODES} return: {ep_return:.2f}")
    finally:
        try:
            trainer.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
