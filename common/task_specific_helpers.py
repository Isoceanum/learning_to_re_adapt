import os

import numpy as np
import yaml
import torch
import math
from torch.utils.data import DataLoader, TensorDataset
from common.planner import make_planner
from common.dynamics_model import DynamicsModel

def load_dataset(dataset_path, eval_split, batch_size, seed):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    data = np.load(dataset_path)

    s = data["s"]
    a = data["a"]
    s_next = data["s_next"]
    episode_id = data["episode_id"]

    unique_episodes = np.unique(episode_id)
    if len(unique_episodes) < 2:
        raise ValueError("Need at least 2 episodes to create train/eval split")

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_episodes)

    eval_count = int(len(unique_episodes) * eval_split)
    if eval_count == 0: eval_count = 1

    eval_ids = set(unique_episodes[-eval_count:])
    eval_mask = np.isin(episode_id, list(eval_ids))
    train_mask = np.logical_not(eval_mask)

    s_train = torch.as_tensor(s[train_mask], dtype=torch.float32)
    a_train = torch.as_tensor(a[train_mask], dtype=torch.float32)
    s_next_train = torch.as_tensor(s_next[train_mask], dtype=torch.float32)

    s_eval = torch.as_tensor(s[eval_mask], dtype=torch.float32)
    a_eval = torch.as_tensor(a[eval_mask], dtype=torch.float32)
    s_next_eval = torch.as_tensor(s_next[eval_mask], dtype=torch.float32)

    train_ds = TensorDataset(s_train, a_train, s_next_train)
    eval_ds = TensorDataset(s_eval, a_eval, s_next_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, eval_loader


def make_planner_from_base_config(pretrained_cfg_path, env, dynamics_fn, device, seed):
    with open(pretrained_cfg_path, "r") as f:
        pretrained_cfg = yaml.safe_load(f)

    planner_cfg = pretrained_cfg["train"]["planner"]
    base_env = getattr(env, "unwrapped", env)
    reward_fn = base_env.get_model_reward_fn()
    return make_planner(planner_cfg, dynamics_fn, reward_fn, env.action_space, device, seed)


def load_pretrained_dynamics_model(pretrained_cfg_path, model_path, obs_dim, action_dim, seed, device):
    with open(pretrained_cfg_path, "r") as f:
        pretrained_cfg = yaml.safe_load(f)

    hidden_sizes = pretrained_cfg["train"]["dynamics_model"]["hidden_sizes"]
    learning_rate = float(pretrained_cfg["train"]["dynamics_model"]["learning_rate"])

    model = DynamicsModel(obs_dim, action_dim, hidden_sizes, learning_rate, seed).to(device)
    model.load_saved_model(model_path)
    model.freeze()
    return model


def eval_policy_rollout(trainer):
    episodes = int(trainer.eval_config["episodes"])
    seeds = trainer.eval_config["seeds"]

    metrics = trainer._evaluate(episodes, seeds)
    elapsed = metrics["elapsed"]
    elapsed_str = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
    reward_mean = metrics["reward_mean"]
    reward_std = metrics["reward_std"]
    ep_len = metrics["episode_length_mean"]
    print(f"eval_rollout: reward={reward_mean:.4f} ± {reward_std:.4f} "f"len={ep_len:.2f} elapsed={elapsed_str}\n")
    return metrics


def compute_cross_task_rmse(trainer):
    dataset_path = trainer.train_config.get("dataset_path")
    if not dataset_path:
        raise AttributeError("Missing dataset_path in YAML")

    eval_split = float(trainer.train_config.get("eval_split"))
    batch_size = int(trainer.train_config.get("batch_size"))

    data_dir = os.path.dirname(dataset_path)
    train_task = os.path.splitext(os.path.basename(dataset_path))[0]
    tasks = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".npz") and f != "nominal.npz"]
    )

    print("\nCross-task RMSE on eval splits")
    task_width = max(len(os.path.splitext(t)[0]) for t in tasks) if tasks else 4
    header = f"{'task':<{task_width}} | {'base_rmse':>10} | {'adapt_rmse':>10} | {'change':>8}"
    print(header)

    csv_lines = ["train_task,eval_task,base_rmse,adapt_rmse"]

    for task_name in tasks:
        task_path = os.path.join(data_dir, task_name)
        task_label = os.path.splitext(task_name)[0]
        _, eval_loader = load_dataset(task_path, eval_split, batch_size, trainer.train_seed)

        base_sse = 0.0
        adapt_sse = 0.0
        count = 0

        trainer.dynamics_model.eval()
        trainer.base_dynamics_model.eval()
        with torch.no_grad():
            for obs_batch, act_batch, next_obs_batch in eval_loader:
                obs_batch = obs_batch.to(trainer.device)
                act_batch = act_batch.to(trainer.device)
                next_obs_batch = next_obs_batch.to(trainer.device)

                base_pred = trainer.base_dynamics_model.predict_next_state(obs_batch, act_batch)
                adapt_pred = trainer.dynamics_model.predict_next_state(obs_batch, act_batch)

                base_err = base_pred - next_obs_batch
                adapt_err = adapt_pred - next_obs_batch

                base_sse += float(torch.sum(base_err ** 2).item())
                adapt_sse += float(torch.sum(adapt_err ** 2).item())
                count += base_err.numel()

        if count == 0:
            print(f"{task_label:<{task_width}} | {'n/a':>10} | {'n/a':>10} | {'n/a':>8}")
            continue

        base_rmse = math.sqrt(base_sse / count)
        adapt_rmse = math.sqrt(adapt_sse / count)

        base_cell = f"{base_rmse:.4f}"
        adapt_cell = f"{adapt_rmse:.4f}"
        if base_rmse == 0:
            change = 0.0
        else:
            change = (base_rmse - adapt_rmse) / base_rmse * 100.0
        change_cell = f"{change:+.1f}%"
        print(f"{task_label:<{task_width}} | {base_cell:>10} | {adapt_cell:>10} | {change_cell:>8}")

        csv_lines.append(f"{train_task},{task_label},{base_rmse:.4f},{adapt_rmse:.4f}")

    print("\nCSV lines (copy/paste):")
    for line in csv_lines:
        print(line)


def eval_epoch_rmse(trainer):
    trainer.dynamics_model.eval()
    trainer.base_dynamics_model.eval()

    eval_sum_mse = 0.0
    eval_count = 0
    base_eval_sum_mse = 0.0
    base_eval_count = 0

    with torch.no_grad():
        for obs_batch, act_batch, next_obs_batch in trainer.eval_dataloader:
            obs_batch = obs_batch.to(trainer.device)
            act_batch = act_batch.to(trainer.device)
            next_obs_batch = next_obs_batch.to(trainer.device)

            pred_next_obs_batch = trainer.dynamics_model.predict_next_state(obs_batch, act_batch)
            loss = torch.mean((pred_next_obs_batch - next_obs_batch) ** 2)
            batch_size = obs_batch.shape[0]
            eval_sum_mse += float(loss.item()) * batch_size
            eval_count += batch_size

            base_pred_next_obs_batch = trainer.base_dynamics_model.predict_next_state(obs_batch, act_batch)
            base_loss = torch.mean((base_pred_next_obs_batch - next_obs_batch) ** 2)
            base_eval_sum_mse += float(base_loss.item()) * batch_size
            base_eval_count += batch_size

    eval_mse = (eval_sum_mse / eval_count) if eval_count > 0 else float("nan")
    base_eval_mse = (base_eval_sum_mse / base_eval_count) if base_eval_count > 0 else float("nan")
    eval_rmse = math.sqrt(eval_mse) if eval_count > 0 else float("nan")
    base_eval_rmse = math.sqrt(base_eval_mse) if base_eval_count > 0 else float("nan")
    return eval_rmse, base_eval_rmse
