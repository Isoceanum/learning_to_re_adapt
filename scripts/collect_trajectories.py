import os
import yaml
import envs  # noqa: F401
import numpy as np
import torch
from tqdm import tqdm

from scripts.run_experiment import _build_trainer

RUN_DIR = "outputs/00_BASE/base_mb"
OUTPUT_DIR = "outputs/2026-03-22/collected_data"
ENV_STEPS = 5000
SEEDS = [1, 2, 3, 4]
MAX_STEPS = 500

TASK_DISTRIBUTION = [
    {"name": "nominal", "type": "nominal"},
    {"name": "cripple_back_right", "type": "cripple", "probability": 1, "candidate_action_indices": [[0, 1]]},
    {"name": "cripple_front_left", "type": "cripple", "probability": 1, "candidate_action_indices": [[2, 3]]},
    {"name": "cripple_front_right", "type": "cripple", "probability": 1, "candidate_action_indices": [[4, 5]]},
    {"name": "cripple_back_left", "type": "cripple", "probability": 1, "candidate_action_indices": [[6, 7]]},
]

def _load_config(run_dir):
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def _get_task_perturbation(task):
    if task.get("type") == "nominal":
        return {}
    return {k: v for k, v in task.items() if k != "name"}


def _collect_task_transitions(trainer, task):
    env_cfg = trainer.environment_config
    perturbation = _get_task_perturbation(task)
    seed = SEEDS[0] if SEEDS else 0
    env = trainer._make_env(env_cfg, perturbation, seed)

    max_episode_length = int(env_cfg["max_episode_length"])
    episode_cap = min(MAX_STEPS, max_episode_length)

    states = []
    actions = []
    next_states = []
    episode_ids = []
    episode_returns = []
    episodes_collected = 0

    steps_collected = 0
    seed_idx = 0

    task_name = task.get("name", "task")
    with tqdm(total=ENV_STEPS, desc=f"collect {task_name}", unit="step") as pbar:
        episode_id = 0
        while steps_collected < ENV_STEPS:
            episode_seed = SEEDS[seed_idx % len(SEEDS)] if SEEDS else 0
            seed_idx += 1

            obs, _ = env.reset(seed=episode_seed)
            done = False
            steps = 0
            episode_return = 0.0

            while not done and steps < episode_cap and steps_collected < ENV_STEPS:
                action = trainer.predict(obs)
                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy()

                next_obs, reward, terminated, truncated, _ = env.step(action)

                states.append(obs)
                actions.append(action)
                next_states.append(next_obs)
                episode_ids.append(episode_id)

                obs = next_obs
                steps += 1
                steps_collected += 1
                episode_return += float(reward)
                done = terminated or truncated
                pbar.update(1)
            if steps > 0:
                episode_returns.append(float(episode_return))
                episodes_collected += 1
                episode_id += 1

    env.close()

    reward_mean = float(np.mean(episode_returns)) if episode_returns else float("nan")
    reward_std = float(np.std(episode_returns)) if episode_returns else float("nan")
    print(f"{task_name}: reward {reward_mean:.2f} ± {reward_std:.2f}, episodes {episodes_collected}")

    return (
        np.asarray(states),
        np.asarray(actions),
        np.asarray(next_states),
        np.asarray(episode_ids),
    )


def main():
    run_dir = os.path.abspath(os.path.expanduser(RUN_DIR))
    config = _load_config(run_dir)

    trainer = _build_trainer(config, run_dir)
    trainer.load(run_dir)

    output_dir = os.path.abspath(os.path.expanduser(OUTPUT_DIR))
    os.makedirs(output_dir, exist_ok=True)

    for task in TASK_DISTRIBUTION:
        s, a, s_next, episode_id = _collect_task_transitions(trainer, task)
        out_path = os.path.join(output_dir, f"{task['name']}.npz")
        np.savez_compressed(out_path, s=s, a=a, s_next=s_next, episode_id=episode_id)
        
    print(f"Saved transitions to {output_dir}")


if __name__ == "__main__":
    main()
