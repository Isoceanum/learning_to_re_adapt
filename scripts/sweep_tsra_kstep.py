import argparse
import copy
import csv
import itertools
import os
import random
import tempfile
import time
from typing import Dict, List

import yaml

from utils.seed import set_seed

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from scripts.run_experiment import _build_trainer


FIXED_PERTURBATION = {
    "type": "cripple",
    "probability": 1,
    "candidate_action_indices": [[6, 7]],
}

FIXED_PLANNER = {
    "type": "cem",
    "horizon": 15,
    "n_candidates": 250,
    "discount": 0.99,
    "num_cem_iters": 4,
    "percent_elites": 0.15,
    "alpha": 0.20,
}


def _create_output_dir(experiment_name: str, explicit_dir: str = None) -> str:
    if explicit_dir:
        out_dir = os.path.abspath(os.path.expanduser(explicit_dir))
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    date = time.strftime("%Y-%m-%d")
    base_path = os.path.join("outputs", date, f"{experiment_name}_sweep")
    path = base_path
    counter = 1
    while os.path.exists(path):
        path = f"{base_path}_{counter}"
        counter += 1
    os.makedirs(path, exist_ok=True)
    return path


def _save_config(config: Dict, output_dir: str):
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _set_nested(cfg: Dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _get_nested(cfg: Dict, dotted_key: str):
    cur = cfg
    for key in dotted_key.split("."):
        cur = cur[key]
    return cur


def _force_fixed_settings(cfg: Dict):
    cfg["algo"] = "tsra_kstep_mppi"
    cfg.setdefault("train", {})
    cfg.setdefault("eval", {})
    cfg["train"]["seed"] = int(cfg["train"].get("seed", 42))
    cfg["train"]["max_episode_length"] = int(cfg["train"].get("max_episode_length", 1000))
    cfg["train"]["eval_interval_steps"] = 0
    cfg["train"]["planner"] = copy.deepcopy(FIXED_PLANNER)
    cfg["train"]["perturbation"] = copy.deepcopy(FIXED_PERTURBATION)
    cfg["eval"]["perturbation"] = copy.deepcopy(FIXED_PERTURBATION)
    cfg.setdefault("experiment_name", "tsra_kstep_ant")
    cfg.setdefault("device", "auto")
    cfg.setdefault("env", "GymAnt-v0")
    cfg.setdefault("exclude_current_positions_from_observation", True)


def _build_sweep_space() -> Dict[str, List]:
    # Large candidate space; runner samples from this space with a max-runs cap.
    return {
        "train.k_horizon": [3, 5, 8],
        "train.lambda_anchor": [0.2, 0.3, 0.4],
        "train.learning_rate": [1e-4, 3e-4, 1e-3],
        "train.batch_size": [128, 256],
        "train.train_epochs": [6, 10],
        "train.residual_adapter.hidden_sizes": [[64, 64], [128, 128]],
        "train.residual_adapter.activation": ["relu", "elu", "tanh"],
        "train.residual_adapter.input_mode": ["sa", "sabase"],
        "train.residual_adapter.bound_type": ["clamp"],
        "train.residual_adapter.bound_max_abs": [0.3, 0.5, 0.8],
        "train.l2_penalty_weight": [0.0, 1e-4],
        "train.iterations": [6, 10],
        "train.steps_per_iteration": [3000, 5000],
    }


def _generate_trials(base_cfg: Dict, max_runs: int, max_total_steps: int, sample_seed: int):
    space = _build_sweep_space()
    keys = list(space.keys())
    value_lists = [space[k] for k in keys]

    all_trials = []
    for values in itertools.product(*value_lists):
        trial = dict(zip(keys, values))
        total_steps = int(trial["train.iterations"]) * int(trial["train.steps_per_iteration"])
        if total_steps > max_total_steps:
            continue
        all_trials.append(trial)

    rng = random.Random(sample_seed)
    rng.shuffle(all_trials)
    selected = all_trials[:max_runs]

    configs = []
    for i, trial in enumerate(selected):
        cfg = copy.deepcopy(base_cfg)
        for k, v in trial.items():
            _set_nested(cfg, k, v)
        cfg["experiment_name"] = f"{base_cfg.get('experiment_name', 'tsra_kstep_ant')}_trial_{i:03d}"
        configs.append((i, trial, cfg))
    return configs


def _evaluate_trainer(trainer, cfg):
    episodes = int(cfg["eval"]["episodes"])
    seeds = cfg["eval"]["seeds"]
    return trainer._evaluate(episodes, seeds)


def _evaluate_base_performance(base_cfg: Dict, sweep_dir: str):
    cfg = copy.deepcopy(base_cfg)
    cfg["experiment_name"] = f"{base_cfg.get('experiment_name', 'tsra_kstep_ant')}_baseline"
    cfg["train"]["iterations"] = 0
    cfg["train"]["steps_per_iteration"] = 0
    cfg["train"]["train_epochs"] = 0

    baseline_dir = os.path.join(sweep_dir, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    _save_config(cfg, baseline_dir)

    set_seed(int(cfg["train"]["seed"]))
    trainer = _build_trainer(cfg, baseline_dir)
    try:
        # Use base-only planner path for clean baseline comparison when available.
        if hasattr(trainer, "base_planner"):
            trainer.planner = trainer.base_planner
        metrics = _evaluate_trainer(trainer, cfg)
        with open(os.path.join(baseline_dir, "baseline_metrics.yaml"), "w") as f:
            yaml.safe_dump(metrics, f, sort_keys=False)
        return metrics
    finally:
        if getattr(trainer, "env", None) is not None:
            trainer.env.close()


def _write_results_csv(path: str, rows: List[Dict]):
    if len(rows) == 0:
        return
    # Stable ordered header for easy spreadsheet consumption.
    fixed_cols = [
        "trial_id",
        "status",
        "output_dir",
        "elapsed_sec",
        "reward_mean",
        "reward_std",
        "forward_progress_mean",
        "forward_progress_std",
        "improvement_abs",
        "improvement_pct",
        "error",
    ]
    param_cols = sorted([k for k in rows[0].keys() if k.startswith("param.")])
    fieldnames = fixed_cols + param_cols

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="TSRA K-step automated hyperparameter sweep.")
    parser.add_argument("config", type=str, help="Base YAML config path")
    parser.add_argument("--max-runs", type=int, default=24, help="Max trial count to run")
    parser.add_argument("--max-total-steps", type=int, default=50000, help="Per-run env step budget upper bound")
    parser.add_argument("--sample-seed", type=int, default=123, help="Sampling seed for trial selection")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional sweep output directory")
    parser.add_argument("--dry-run", action="store_true", help="Only print selected trial configs")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    _force_fixed_settings(base_cfg)

    sweep_dir = _create_output_dir(base_cfg.get("experiment_name", "tsra_kstep_ant"), args.output_dir)
    print(f"[sweep] output_dir={sweep_dir}")
    print("[sweep] fixed perturbation=cripple prob=1 candidate_action_indices=[[6,7]]")
    print("[sweep] fixed planner=cem horizon=15 n_candidates=250 discount=0.99 num_cem_iters=4 percent_elites=0.15 alpha=0.20")
    print(f"[sweep] per-run step budget <= {args.max_total_steps}")

    trial_cfgs = _generate_trials(base_cfg, args.max_runs, args.max_total_steps, args.sample_seed)
    print(f"[sweep] selected_trials={len(trial_cfgs)}")

    if args.dry_run:
        for i, trial, _ in trial_cfgs:
            total_steps = int(trial["train.iterations"]) * int(trial["train.steps_per_iteration"])
            print(f"trial={i:03d} total_steps={total_steps} params={trial}")
        return

    baseline_metrics = _evaluate_base_performance(base_cfg, sweep_dir)
    baseline_reward = float(baseline_metrics["reward_mean"])
    print(f"[baseline] reward_mean={baseline_reward:.4f}")

    results = []
    best_row = None

    for i, trial, cfg in trial_cfgs:
        trial_dir = os.path.join(sweep_dir, f"trial_{i:03d}")
        os.makedirs(trial_dir, exist_ok=True)
        _save_config(cfg, trial_dir)

        print(f"\n[sweep] trial={i:03d} start params={trial}")
        start = time.time()
        row = {
            "trial_id": i,
            "status": "ok",
            "output_dir": trial_dir,
            "elapsed_sec": float("nan"),
            "reward_mean": float("nan"),
            "reward_std": float("nan"),
            "forward_progress_mean": float("nan"),
            "forward_progress_std": float("nan"),
            "improvement_abs": float("nan"),
            "improvement_pct": float("nan"),
            "error": "",
        }
        for k, v in trial.items():
            row[f"param.{k}"] = v

        trainer = None
        try:
            set_seed(int(cfg["train"]["seed"]))
            trainer = _build_trainer(cfg, trial_dir)
            trainer.train()
            trainer.save()
            metrics = _evaluate_trainer(trainer, cfg)

            reward_mean = float(metrics["reward_mean"])
            improvement_abs = reward_mean - baseline_reward
            improvement_pct = (100.0 * improvement_abs / abs(baseline_reward)) if abs(baseline_reward) > 1e-8 else float("nan")

            row["elapsed_sec"] = time.time() - start
            row["reward_mean"] = reward_mean
            row["reward_std"] = float(metrics["reward_std"])
            row["forward_progress_mean"] = float(metrics["forward_progress_mean"])
            row["forward_progress_std"] = float(metrics["forward_progress_std"])
            row["improvement_abs"] = improvement_abs
            row["improvement_pct"] = improvement_pct

            print(
                f"[sweep] trial={i:03d} done reward_mean={reward_mean:.4f} "
                f"improve={improvement_abs:+.4f} ({improvement_pct:+.2f}%) elapsed={row['elapsed_sec']:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001
            row["status"] = "failed"
            row["elapsed_sec"] = time.time() - start
            row["error"] = repr(exc)
            print(f"[sweep] trial={i:03d} failed error={row['error']}")
        finally:
            if trainer is not None and getattr(trainer, "env", None) is not None:
                trainer.env.close()

        results.append(row)
        if row["status"] == "ok":
            if best_row is None or row["improvement_abs"] > best_row["improvement_abs"]:
                best_row = row

        _write_results_csv(os.path.join(sweep_dir, "sweep_results.csv"), results)

    summary = {
        "sweep_dir": sweep_dir,
        "baseline_reward_mean": baseline_reward,
        "num_trials": len(results),
        "num_success": sum(1 for r in results if r["status"] == "ok"),
        "num_failed": sum(1 for r in results if r["status"] != "ok"),
        "best_trial": best_row,
    }

    with open(os.path.join(sweep_dir, "sweep_summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("\n[sweep] completed")
    print(f"[sweep] successes={summary['num_success']} failures={summary['num_failed']}")
    if best_row is None:
        print("[sweep] no successful trial")
        return

    print(
        f"[best] trial={best_row['trial_id']:03d} reward_mean={best_row['reward_mean']:.4f} "
        f"improve={best_row['improvement_abs']:+.4f} ({best_row['improvement_pct']:+.2f}%)"
    )
    print(f"[best] output_dir={best_row['output_dir']}")


if __name__ == "__main__":
    main()
