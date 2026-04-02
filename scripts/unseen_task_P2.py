import copy
import os
import yaml
import time
import numpy as np
from tqdm import tqdm

# Ensure custom envs are registered
import envs  # noqa: F401

from scripts.run_experiment import _build_trainer
from utils.seed import set_seed


RUNS = [
    {
        "path": "/Users/isoceanum/Projects/l2ra/learning_to_re_adapt/outputs/2026-03-28-ex3/mb_mpc_10k",
        "label": "mb_mpc",
    },
    {
        "path": "/Users/isoceanum/Projects/l2ra/learning_to_re_adapt/outputs/2026-03-28-ex3/grbal_mpc_10k",
        "label": "grbal",
    },
    {
        "path": "/Users/isoceanum/Projects/l2ra/learning_to_re_adapt/outputs/2026-03-28-ex3/meta_lora_10k",
        "label": "meta_lora",
    },
    {
        "path": "/Users/isoceanum/Projects/l2ra/learning_to_re_adapt/outputs/2026-03-28-ex3/meta_bitfit_10k",
        "label": "meta_bitfit",
    },
]

TASK_NAME = "cripple leg"

PERTURBATION = {
    "type": "cripple",
    "probability": 1,
    "candidate_action_indices": [[0, 1]],
}

EPISODES = 4
SEEDS = [42]


def _load_config(run_dir):
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _evaluate_with_progress(trainer, episodes, seeds, desc):
    all_rewards = []
    episode_lengths = []
    eval_start_time = time.time()
    eval_perturbation_config = trainer.eval_config.get("perturbation", {})
    max_episode_length = int(trainer.environment_config["max_episode_length"])

    total = len(seeds) * episodes
    with tqdm(total=total, desc=desc, unit="ep") as pbar:
        for seed in seeds:
            set_seed(seed)
            eval_env = trainer._make_env(trainer.environment_config, eval_perturbation_config, seed)

            for _ in range(episodes):
                # Reset planner RNG so identical seed yields identical rollout.
                if hasattr(trainer, "planner") and hasattr(trainer.planner, "gen"):
                    trainer.planner.gen.manual_seed(int(seed))
                trajectory, metrics = trainer._rollout_episode(eval_env, 1, max_episode_length)
                episode_obs, _, _ = trajectory
                all_rewards.append(metrics["episode_return"])
                episode_lengths.append(int(len(episode_obs)))
                trainer._reset_episode_state()
                pbar.update(1)

            eval_env.close()

    return {
        "reward_mean": float(np.mean(all_rewards)),
        "reward_std": float(np.std(all_rewards)),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "elapsed": time.time() - eval_start_time,
    }


def _evaluate_run(run_dir, label):
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    config = _load_config(run_dir)

    trainer = _build_trainer(config, run_dir)
    trainer.load(run_dir)

    eval_cfg = copy.deepcopy(config.get("eval", {}))
    eval_cfg["perturbation"] = PERTURBATION
    trainer.eval_config = eval_cfg

    metrics = _evaluate_with_progress(
        trainer,
        int(EPISODES),
        [int(s) for s in SEEDS],
        desc=f"{label} eval",
    )
    mean_return = float(metrics.get("reward_mean", float("nan")))
    std_return = float(metrics.get("reward_std", float("nan")))

    print(f"{TASK_NAME},{label},{mean_return:.4f},{std_return:.4f}")


def main():
    for run in RUNS:
        _evaluate_run(run["path"], run["label"])


if __name__ == "__main__":
    main()
