import argparse
import os
import yaml

# Ensure custom envs are registered
import envs  # noqa: F401
import gymnasium as gym

from perturbations.perturbation_factory import resolve_perturbation_env
from scripts.run_experiment import _build_trainer

def main():
    parser = argparse.ArgumentParser(description="Render a trained RL policy from a run directory")
    parser.add_argument("run_dir", type=str, help="Path to a trained run directory (contains config.yaml, model.zip)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to render")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions for rendering")
    args = parser.parse_args()

    run_dir = args.run_dir
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # Build trainer bound to this run directory
    trainer = _build_trainer(config, run_dir)
    # Recreate env in render mode
    env_id = config.get("env")
    
    trainer.env = gym.make(env_id, exclude_current_positions_from_observation=False)
    trainer.env = resolve_perturbation_env(trainer.env, trainer.eval_config, 0)

    # Load trained weights (let each trainer resolve its default model filename)
    trainer.load(run_dir)

    # Rollout episodes
    episodes = args.episodes
    deterministic = bool(args.deterministic)
    print(f"🎥 Rendering {episodes} episode(s) with deterministic={deterministic}")

    try:
        for ep in range(episodes):
            obs, _ = trainer.env.reset()
            done = False
            ep_return = 0.0
            while not done:
                action = trainer.predict(obs)
                obs, reward, terminated, truncated, _ = trainer.env.step(action)
                done = terminated or truncated
                ep_return += float(reward)
            print(f"Episode {ep+1}/{episodes} return: {ep_return:.2f}")
    finally:
        try:
            trainer.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

