import os
import yaml
import numpy as np
import envs  # noqa: F401

from scripts.run_experiment import _build_trainer
from evaluation.model_error import compute_k_step_rmse_for_episode

RUN_DIR = "outputs/2026-03-05/mb_mpc_robust_sweep_08_p20_low_noise"
EPISODES = 2
SEEDS = [1, 2, 3, 4]
MAX_STEPS = 1000

TASK_DISTRIBUTION = [
    {"name": "nominal", "type": "nominal"},
    {"name": "cripple_01", "type": "cripple", "probability": 1, "candidate_action_indices": [[0, 1]]},
    {"name": "cripple_23", "type": "cripple", "probability": 1, "candidate_action_indices": [[2, 3]]},
    {"name": "cripple_45", "type": "cripple", "probability": 1, "candidate_action_indices": [[4, 5]]},
    {"name": "cripple_67", "type": "cripple", "probability": 1, "candidate_action_indices": [[6, 7]]},
]


def _load_config(run_dir):
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _evaluate_task(trainer, task, seeds, max_steps):
    eval_cfg = trainer.eval_config.copy() if isinstance(trainer.eval_config, dict) else {}

    if task.get("type") == "nominal":
        eval_cfg.pop("perturbation", None)
    else:
        perturbation = {k: v for k, v in task.items() if k != "name"}
        eval_cfg["perturbation"] = perturbation

    trainer.set_eval_config(eval_cfg)
    trainer._reset_eval_planner()

    rmse_values = []

    for seed in seeds:
        obs, _ = trainer.env.reset(seed=seed)
        done = False
        steps = 0
        episode_transitions = []

        while not done and steps < max_steps:
            action = trainer.predict(obs)
            next_obs, _, terminated, truncated, _ = trainer.env.step(action)
            episode_transitions.append((obs, action, next_obs))
            obs = next_obs
            done = terminated or truncated
            steps += 1

        device = next(trainer.dynamics_model.parameters()).device
        rmse_by_k = compute_k_step_rmse_for_episode(episode_transitions, trainer.dynamics_model, [1], device)
        rmse_values.append(rmse_by_k[1])

    mean_rmse = float(np.mean(rmse_values)) if rmse_values else float("nan")
    std_rmse = float(np.std(rmse_values)) if rmse_values else float("nan")
    return mean_rmse, std_rmse


def main():
    run_dir = os.path.abspath(os.path.expanduser(RUN_DIR))
    config = _load_config(run_dir)

    trainer = _build_trainer(config, run_dir)
    trainer.load(run_dir)

    model_name = os.path.basename(os.path.normpath(run_dir))

    headers = [task["name"] for task in TASK_DISTRIBUTION]
    results = []

    for task in TASK_DISTRIBUTION:
        mean_rmse, std_rmse = _evaluate_task(trainer, task, SEEDS, MAX_STEPS)
        results.append(f"{mean_rmse:.4f} ± {std_rmse:.4f}")

    header_line = "task".ljust(14) + "| " + " | ".join(name.ljust(18) for name in headers)
    value_line = model_name.ljust(14) + "| " + " | ".join(val.ljust(18) for val in results)

    print(header_line)
    print(value_line)


if __name__ == "__main__":
    main()
