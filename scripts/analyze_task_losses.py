import os
from collections import defaultdict

import numpy as np
import torch
import yaml

from algorithms.full_param_mem.trainer import FullParamMemoryTrainer
from utils.seed import set_seed


RUN_DIR = "/Users/isoceanum/Projects/l2ra/learning_to_re_adapt/outputs/2026-04-13/full_param_mem"
OVERRIDES = {
    "train": {
        "seed": 42,
        "use_memory": False,
        "use_online_adaptation": False,
        "support_window_size": 16,
        "memory_retrieval_abs_improvement_threshold": 0.00050,
        "memory_retrieval_rel_improvement_threshold": 0.00500,
        "memory_retrieval_winner_margin_abs_threshold": 0.00005,
        "memory_retrieval_winner_margin_rel_threshold": 0.00100,
        "memory_max_entries": 8,
        "memory_commit_abs_improvement_threshold": 0.00005,
        "memory_commit_rel_improvement_threshold": 0.00100,
        "slow_outer_learning_rate": 0.001,
        "memory_retrieval_min_transitions": 32,
    },
    "eval": {
        "episodes": 20,
        "seeds": [42],
        "perturbation": {
            "type": "cripple",
            "probability": 1,
            "candidate_action_indices": [[0, 1], [2, 3], [4, 5], [6, 7]],
        },
    },
}


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _episode_prediction_metrics(trainer, episode_obs, episode_act, episode_next_obs):
    device = trainer.device
    obs = torch.as_tensor(np.stack(episode_obs, axis=0), dtype=torch.float32, device=device)
    act = torch.as_tensor(np.stack(episode_act, axis=0), dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(np.stack(episode_next_obs, axis=0), dtype=torch.float32, device=device)
    delta = next_obs - obs

    theta = trainer.dynamics_model.get_parameter_dict()
    with torch.no_grad():
        norm_loss_base = float(
            trainer.dynamics_model.compute_loss_with_parameters(obs, act, delta, theta).item()
        )
        pred_next_base = trainer.dynamics_model.predict_next_state_with_parameters(obs, act, theta)
        raw_mse_base = float(torch.mean((pred_next_base - next_obs) ** 2).item())

    support_n = int(trainer.support_window_size)
    if len(episode_obs) >= support_n:
        support_obs = obs[:support_n]
        support_act = act[:support_n]
        support_next_obs = next_obs[:support_n]
        adapted_theta = trainer.dynamics_model.compute_adapted_parameters_step(
            theta,
            support_obs,
            support_act,
            support_next_obs,
            float(trainer.inner_learning_rate),
            create_graph=False,
        )
        with torch.no_grad():
            norm_loss_adapted = float(
                trainer.dynamics_model.compute_loss_with_parameters(obs, act, delta, adapted_theta).item()
            )
            pred_next_adapted = trainer.dynamics_model.predict_next_state_with_parameters(obs, act, adapted_theta)
            raw_mse_adapted = float(torch.mean((pred_next_adapted - next_obs) ** 2).item())
    else:
        norm_loss_adapted = np.nan
        raw_mse_adapted = np.nan

    return {
        "norm_loss_base": norm_loss_base,
        "raw_mse_base": raw_mse_base,
        "norm_loss_adapted": norm_loss_adapted,
        "raw_mse_adapted": raw_mse_adapted,
    }


def _print_task_summary(task_stats):
    print("\n[analysis] ===== Per-Task Prediction/Loss Summary =====")
    if not task_stats:
        print("[analysis] no data")
        print("[analysis] ==========================================")
        return

    for task_name in sorted(task_stats.keys()):
        rows = task_stats[task_name]
        rewards = np.array([r["reward"] for r in rows], dtype=np.float64)
        norm_base = np.array([r["norm_loss_base"] for r in rows], dtype=np.float64)
        norm_adapted = np.array([r["norm_loss_adapted"] for r in rows], dtype=np.float64)
        raw_base = np.array([r["raw_mse_base"] for r in rows], dtype=np.float64)
        raw_adapted = np.array([r["raw_mse_adapted"] for r in rows], dtype=np.float64)

        print(f"[analysis] task={task_name} n={len(rows)}")
        print(f"[analysis]   reward_mean={rewards.mean():.4f} reward_std={rewards.std():.4f}")
        print(f"[analysis]   norm_loss_base_mean={norm_base.mean():.8f} norm_loss_base_std={norm_base.std():.8f}")
        print(f"[analysis]   norm_loss_adapted_mean={np.nanmean(norm_adapted):.8f} norm_loss_adapted_std={np.nanstd(norm_adapted):.8f}")
        print(f"[analysis]   raw_mse_base_mean={raw_base.mean():.8f} raw_mse_base_std={raw_base.std():.8f}")
        print(f"[analysis]   raw_mse_adapted_mean={np.nanmean(raw_adapted):.8f} raw_mse_adapted_std={np.nanstd(raw_adapted):.8f}")

    print("[analysis] ==========================================")


def main():
    config_path = os.path.join(RUN_DIR, "config.yaml")
    model_path = os.path.join(RUN_DIR, "model.pt")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = _deep_update(config, OVERRIDES)

    set_seed(int(config["train"]["seed"]))
    trainer = FullParamMemoryTrainer(config, RUN_DIR)
    trainer.load(RUN_DIR)

    episodes = int(config["eval"]["episodes"])
    seeds = list(config["eval"]["seeds"])
    eval_perturbation_config = config["eval"].get("perturbation", {})
    max_episode_length = int(config["environment"]["max_episode_length"])

    print(
        f"Analyzing model loss by task with memory disabled: "
        f"episodes={episodes}, seeds={seeds}, "
        f"candidate_action_indices={eval_perturbation_config.get('candidate_action_indices')}"
    )

    task_stats = defaultdict(list)

    for seed in seeds:
        set_seed(seed)
        eval_env = trainer._make_env(trainer.environment_config, eval_perturbation_config, seed)
        for episode in range(episodes):
            # Manual trainer-state reset without invoking memory permanent-step logging.
            trainer.current_task = None
            trainer.recent_transitions.clear()
            trainer.last_obs = None
            trainer.last_action = None
            trainer._memory_selected_this_episode = False
            trainer._episode_memory_id = None
            trainer._episode_params = None
            trainer._episode_retrieval_reason = None
            trainer._episode_retrieval_improvement = None

            trajectory, rollout_metrics = trainer._rollout_episode(eval_env, 1, max_episode_length)
            episode_obs, episode_act, episode_next_obs = trajectory
            task_name = str(trainer.current_task)
            reward = float(rollout_metrics["episode_return"])
            ep_len = int(len(episode_obs))
            pred_metrics = _episode_prediction_metrics(trainer, episode_obs, episode_act, episode_next_obs)

            task_stats[task_name].append(
                {
                    "seed": seed,
                    "episode": episode + 1,
                    "reward": reward,
                    **pred_metrics,
                }
            )

            print(
                f"[analysis] seed={seed} episode={episode+1}/{episodes} task={task_name} len={ep_len} "
                f"reward={reward:.4f} norm_loss_base={pred_metrics['norm_loss_base']:.8f} "
                f"norm_loss_adapted={pred_metrics['norm_loss_adapted']:.8f} "
                f"raw_mse_base={pred_metrics['raw_mse_base']:.8f} raw_mse_adapted={pred_metrics['raw_mse_adapted']:.8f}"
            )

        eval_env.close()

    _print_task_summary(task_stats)


if __name__ == "__main__":
    main()
