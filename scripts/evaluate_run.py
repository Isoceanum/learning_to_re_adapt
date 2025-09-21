"""Evaluate a saved run using the perturbation-aware trainer pipeline."""

import argparse
import os
import yaml


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained run with perturbations enabled")
    parser.add_argument("run_dir", type=str, help="Path to the saved training run directory")
    args = parser.parse_args()

    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")

    with open(cfg_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    from scripts.train import _build_trainer

    trainer = _build_trainer(config, run_dir)
    try:
        trainer.load(run_dir)
    except AttributeError:
        # Some trainers may not implement load; ignore silently
        pass

    print(f"Evaluating run at {run_dir}")
    trainer.evaluate()


if __name__ == "__main__":
    main()
