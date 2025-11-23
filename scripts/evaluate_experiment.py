import argparse
import os
import yaml

# Ensure custom envs are registered
import envs  # noqa: F401
import gymnasium as gym

from scripts.run_experiment import _build_trainer

def main():
    parser = argparse.ArgumentParser(description="None")
    parser.add_argument("run_directory", type=str, help="Path to a trained run directory (contains config.yaml, model.zip)")
    parser.add_argument("eval_yaml", type=str, help="Path to a trained run directory (contains config.yaml, model.zip)")
    args = parser.parse_args()

    run_directory = args.run_directory
    
    # 1 Build traning and load model
    
    # 2 Read eval config if provided or use the one inside base config and build a solid eval env based on it
    
    # run eval and collect pertubation aware metrics and generate csv that can later be used ot make plots 


    cfg_path = os.path.join(run_directory, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found in {run_directory}")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # Build trainer bound to this run directory
    trainer = _build_trainer(config, run_directory)

    # Recreate env in render mode
    env_id = config.get("env")
    
    trainer.env = gym.make(env_id, render_mode="human", exclude_current_positions_from_observation=False)

    # Load trained weights (let each trainer resolve its default model filename)
    trainer.load(run_directory)

    # Rollout episodes
    episodes = args.episodes
    deterministic = bool(args.deterministic)
    print(f" Rendering {episodes} episode(s) with deterministic={deterministic}")

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

