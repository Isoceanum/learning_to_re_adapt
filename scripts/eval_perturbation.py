import copy
import os
import time
import yaml

# Ensure custom envs are registered
import envs  # noqa: F401

from scripts.run_experiment import _build_trainer


RUN_DIR = "outputs/2026-03-05/mb_mpc_robust_sweep_08_p20_low_noise"
EPISODES = 3
SEEDS = [1,2,3,4]

PERTURBATIONS = [
    {"type": "cripple", "probability": 1, "candidate_action_indices": [[0,1]]},
    {"type": "cripple", "probability": 1, "candidate_action_indices": [[2,3]]},
    {"type": "cripple", "probability": 1, "candidate_action_indices": [[4,5]]},
    {"type": "cripple", "probability": 1, "candidate_action_indices": [[6,7]]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [0], "range": [0.0, 0.0]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [1], "range": [0.0, 0.0]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [2], "range": [0.0, 0.0]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [3], "range": [0.0, 0.0]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [4], "range": [0.0, 0.0]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [5], "range": [0.0, 0.0]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [6], "range": [0.0, 0.0]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [7], "range": [0.0, 0.0]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [0], "range": [0.5, 0.5]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [1], "range": [0.5, 0.5]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [2], "range": [0.5, 0.5]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [3], "range": [0.5, 0.5]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [4], "range": [0.5, 0.5]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [5], "range": [0.5, 0.5]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [6], "range": [0.5, 0.5]},
    # {"type": "action_scaling", "probability": 1, "candidate_action_indices": [7], "range": [0.5, 0.5]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [0, 1]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [0, 2]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [0, 3]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [0, 4]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [0, 5]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [0, 6]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [0, 7]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [1, 2]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [1, 3]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [1, 4]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [1, 5]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [1, 6]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [1, 7]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [2, 3]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [2, 4]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [2, 5]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [2, 6]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [2, 7]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [3, 4]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [3, 5]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [3, 6]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [3, 7]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [4, 5]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [4, 6]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [4, 7]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [5, 6]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [5, 7]},
    # {"type": "action_swap", "probability": 1, "swap_pairs": [6, 7]},
]


# -----------------------------
# Helpers
# -----------------------------

def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _resolve_dynamics_model_path(config, run_dir):
    train_cfg = config.get("train", {})
    orig_path = train_cfg.get("dynamics_model_path")
    if not orig_path:
        return
    if os.path.isdir(orig_path):
        return

    repo_root = _repo_root()
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


def _resolve_pretrained_dynamics_paths(config, run_dir):
    train_cfg = config.get("train", {})
    pre_cfg = train_cfg.get("pretrained_dynamics_model")
    if not isinstance(pre_cfg, dict):
        return

    repo_root = _repo_root()

    for key in ("model_path", "config_path"):
        orig_path = pre_cfg.get(key)
        if not orig_path:
            continue
        if os.path.exists(orig_path):
            continue

        candidates = []
        if "outputs" in orig_path:
            suffix = orig_path[orig_path.index("outputs") :]
            candidates.append(os.path.join(repo_root, suffix))

        if not os.path.isabs(orig_path):
            candidates.append(os.path.abspath(os.path.join(run_dir, orig_path)))
            candidates.append(os.path.abspath(os.path.join(repo_root, orig_path)))

        for cand in candidates:
            if os.path.exists(cand):
                pre_cfg[key] = cand
                print(f"Resolved pretrained_dynamics_model.{key} -> {cand}")
                break
        else:
            print(f"Warning: pretrained_dynamics_model.{key} not found locally: {orig_path}")


def _format_perturbation_label(perturbation):
    if not perturbation:
        return "none"
    parts = []
    for key, value in perturbation.items():
        parts.append(f"{key}: {value}")
    return " ".join(parts)


def _format_pct_change(value, baseline):
    if baseline == 0:
        return "n/a"
    pct = (value - baseline) / baseline * 100.0
    return f"{pct:+.2f}%"


def _resolve_run_dir():
    return os.path.abspath(os.path.expanduser(RUN_DIR))


def _resolve_episodes_and_seeds(_config):
    episodes = int(EPISODES)
    seeds = [int(s) for s in SEEDS]
    return episodes, seeds


def main():
    run_dir = _resolve_run_dir()
    if not run_dir:
        raise FileNotFoundError("No run directory found. Set RUN_DIR or add runs under outputs/.")

    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    _resolve_dynamics_model_path(config, run_dir)
    _resolve_pretrained_dynamics_paths(config, run_dir)

    trainer = _build_trainer(config, run_dir)
    trainer.load(run_dir)

    episodes, seeds = _resolve_episodes_and_seeds(config)
    base_eval_cfg = config.get("eval", {})

    print("Eval perturbation")
    print(f"run_dir: {run_dir}")
    print(f"episodes: {episodes}")
    print(f"seeds: {seeds}")
    print(f"perturbations: {len(PERTURBATIONS)}")

    results = []
    start_time = time.time()

    # Nominal (no perturbation) baseline
    nominal_cfg = copy.deepcopy(base_eval_cfg)
    nominal_cfg.pop("perturbation", None)
    trainer.set_eval_config(nominal_cfg)
    trainer._reset_eval_planner()

    print("\n[nominal] no perturbation")
    nominal_metrics = trainer._evaluate(episodes, seeds)
    results.append(("nominal", nominal_metrics))
    print(
        "reward: {:.4f} ± {:.4f} forward_progress: {:.4f} ± {:.4f}".format(
            nominal_metrics["reward_mean"],
            nominal_metrics["reward_std"],
            nominal_metrics["forward_progress_mean"],
            nominal_metrics["forward_progress_std"],
        )
    )

    baseline_reward = nominal_metrics["reward_mean"]
    baseline_forward = nominal_metrics["forward_progress_mean"]

    for idx, perturbation in enumerate(PERTURBATIONS, start=1):
        eval_cfg = copy.deepcopy(base_eval_cfg)
        eval_cfg["perturbation"] = perturbation
        trainer.set_eval_config(eval_cfg)
        trainer._reset_eval_planner()

        label = _format_perturbation_label(perturbation)
        print(f"\n[{idx}/{len(PERTURBATIONS)}] {label}")

        metrics = trainer._evaluate(episodes, seeds)
        results.append((label, metrics))

        print(
            "reward: {:.4f} ± {:.4f} forward_progress: {:.4f} ± {:.4f}".format(
                metrics["reward_mean"],
                metrics["reward_std"],
                metrics["forward_progress_mean"],
                metrics["forward_progress_std"],
            )
        )
        print(
            "delta vs nominal | reward: {} forward_progress: {}".format(
                _format_pct_change(metrics["reward_mean"], baseline_reward),
                _format_pct_change(metrics["forward_progress_mean"], baseline_forward),
            )
        )

    elapsed = time.time() - start_time
    elapsed_str = f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"

    print("\nSummary")
    print(f"elapsed: {elapsed_str}")
    for label, metrics in results:
        print(label)
        if label == "nominal":
            print(
                "reward: {rm:.2f} ± {rs:.2f} fp: {fm:.2f} ± {fs:.2f}".format(
                    rm=metrics["reward_mean"],
                    rs=metrics["reward_std"],
                    fm=metrics["forward_progress_mean"],
                    fs=metrics["forward_progress_std"],
                )
            )
        else:
            delta_reward = _format_pct_change(metrics["reward_mean"], baseline_reward)
            delta_forward = _format_pct_change(metrics["forward_progress_mean"], baseline_forward)
            print(
                "reward: {rm:.2f} ± {rs:.2f} fp: {fm:.2f} ± {fs:.2f} delta vs nominal | reward: {dr} fp: {df}".format(
                    rm=metrics["reward_mean"],
                    rs=metrics["reward_std"],
                    fm=metrics["forward_progress_mean"],
                    fs=metrics["forward_progress_std"],
                    dr=delta_reward,
                    df=delta_forward,
                )
            )
        print()


if __name__ == "__main__":
    main()
