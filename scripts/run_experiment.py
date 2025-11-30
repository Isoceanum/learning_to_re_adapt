# scripts/train.py
import argparse
import yaml
import os
import time
import shutil

from algorithms.ppo.trainer import PPOTrainer
from algorithms.mb_mpc_fidelity.trainer import MBMPCFidelityTrainer
from utils.seed import set_seed

#algorithms
def _build_trainer(config, output_dir):
    algo = config.get("algo").lower()
    
    if algo == "ppo":
        return PPOTrainer(config, output_dir)

    elif algo == "mb_mpc_fidelity":
        return MBMPCFidelityTrainer(config, output_dir)
        
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
    parser.add_argument("--output-dir", type=str, default=None, help=("Optional output directory to use directly"))
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
            
    if args.output_dir:
        out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
        # Guard against specifying a path that is an existing file
        if os.path.isfile(out_dir):
            raise ValueError(f"--output-dir points to a file, not a directory: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = _create_output_dir(config.get("experiment_name"))
        
    _save_config(args.config, out_dir)
        
    set_seed(int(config["train"]["seed"]))
    trainer = _build_trainer(config, out_dir)
    trainer.train()
    trainer.save()
    trainer.evaluate()
    
    
if __name__ == "__main__":
    main()
