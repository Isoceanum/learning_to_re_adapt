# scripts/train.py
import argparse
import yaml
import os
import time
import shutil

from algorithms.ppo.ppo_trainer import PPOTrainer

def _build_trainer(config, output_dir):
    algo = config.get("algo").lower()

    if algo == "ppo":
        return PPOTrainer(config, output_dir)
    elif algo == "grbal":
        raise NotImplementedError("GrBAL trainer not implemented yet")
    elif algo == "rebal":
        raise NotImplementedError("ReBAL trainer not implemented yet")
    elif algo == "mb_mpc":
        raise NotImplementedError("MB-MPC trainer not implemented yet")
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

def _create_output_dir(experiment_name):
    date = time.strftime("%Y-%m-%d")
    base_path = os.path.join("outputs", date, experiment_name)
    path = base_path
    counter = 1

    while os.path.exists(path):
        path = f"{base_path}_{counter}"
        counter += 1

    os.makedirs(path, exist_ok=True)
    return path

def _save_config(configName, output_dir):
    dest_path = os.path.join(output_dir, "config.yaml")
    shutil.copy(configName, dest_path)
    
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train an RL agent from a YAML config.")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
            
    out_dir = _create_output_dir(config.get("experiment_name"))
    trainer = _build_trainer(config, out_dir)
    trainer.train()
    trainer.evaluate(episodes=3)
    
    _save_config(args.config, out_dir)

if __name__ == "__main__":
    main()
