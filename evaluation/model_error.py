
import time
from utils.seed import seed_env, set_seed
import torch
import math
import numpy as np


def compute_k_step_rmse_for_episode(episode_transitions, model, k_list, device):
    k_max = max(k_list)
    num_transitions = len(episode_transitions)

    sum_mse_by_k = {k: 0.0 for k in k_list}
    count_by_k = {k: 0 for k in k_list}

    for start in range(0, num_transitions - k_max + 1):
        start_obs = episode_transitions[start][0]
        pred_obs = torch.as_tensor(start_obs, dtype=torch.float32, device=device).unsqueeze(0)

        for k in range(1, k_max + 1):
            action = episode_transitions[start + k - 1][1]
            true_next_obs = episode_transitions[start + k - 1][2]

            action = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            true_next_obs = torch.as_tensor(true_next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            pred_next_obs = model.predict_next_state(pred_obs, action)

            if k in k_list:
                err = pred_next_obs - true_next_obs
                mse = (err ** 2).mean().item()
                sum_mse_by_k[k] += mse
                count_by_k[k] += 1
            pred_obs = pred_next_obs
            
    mse_by_k = {k: (sum_mse_by_k[k] / max(1, count_by_k[k])) for k in k_list}
    rmse_by_k = {k: math.sqrt(mse_by_k[k]) for k in k_list}
    return rmse_by_k
    
def compute_top_rmse_by_dim_for_episode(episode_transitions, model, max_k, device, top_n):
    k_targets = (1, max_k)
    num_transitions = len(episode_transitions)

    # per-dim SSE accumulators (initialized lazily once we know obs_dim)
    sse_by_k = {k: None for k in k_targets}
    count_by_k = {k: 0 for k in k_targets}

    for start in range(0, num_transitions - max_k + 1):
        start_obs = episode_transitions[start][0]
        pred_obs = torch.as_tensor(start_obs, dtype=torch.float32, device=device).unsqueeze(0)

        for k in range(1, max_k + 1):
            action = episode_transitions[start + k - 1][1]
            true_next_obs = episode_transitions[start + k - 1][2]

            action = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            true_next_obs = torch.as_tensor(true_next_obs, dtype=torch.float32, device=device).unsqueeze(0)

            pred_next_obs = model.predict_next_state(pred_obs, action)

            if k in sse_by_k:
                err = (pred_next_obs - true_next_obs).squeeze(0)  # (obs_dim,)
                if sse_by_k[k] is None:
                    sse_by_k[k] = torch.zeros_like(err)
                sse_by_k[k] += err.pow(2)
                count_by_k[k] += 1

            pred_obs = pred_next_obs

    top_by_k = {}
    for k in k_targets:
        if sse_by_k[k] is None or count_by_k[k] == 0:
            top_by_k[k] = []
            continue

        rmse_per_dim = torch.sqrt(sse_by_k[k] / count_by_k[k])  # (obs_dim,)
        vals, idxs = torch.topk(rmse_per_dim, k=min(top_n, rmse_per_dim.numel()))
        top_by_k[k] = [(int(i.item()), float(v.item())) for v, i in zip(vals, idxs)]

    return top_by_k

def evaluate_dynamics_rmse(make_env, policy, model, seeds, k_list, max_steps):
    eval_start_time = time.time()
    rmse_values_by_k = {k: [] for k in k_list}
    episode_rewards = []
    episode_forward_progresses = []
    top_dim_counts_k1 = {}
    for seed in seeds:
        print(f"\n[seed {seed}]")
        
        set_seed(seed)
        env = make_env(seed=seed)

        obs, _ = env.reset(seed=seed)
        done = False
        steps = 0
        episode_transitions = []
        ep_reward = 0.0
        com_x_start = None
        last_com_x = None

        while not done and steps < max_steps:  
            action = policy(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_transitions.append((obs, action, next_obs))
            obs = next_obs
            ep_reward += float(reward)
            done = terminated or truncated
            steps += 1
            
            if com_x_start is None:
                com_x_start = float(info["x_position"])
            last_com_x = float(info["x_position"])
            
        forward_progress = last_com_x - com_x_start if (com_x_start is not None and last_com_x is not None) else 0.0
            
        episode_rewards.append(ep_reward)
        episode_forward_progresses.append(forward_progress)
        env.close()
        device = next(model.parameters()).device
        rmse_by_k = compute_k_step_rmse_for_episode(episode_transitions, model, k_list, device)
        for k in k_list: rmse_values_by_k[k].append(rmse_by_k[k])
        
        print("RMSE:", " | ".join([f"k-{k} {rmse_by_k[k]:.4f}" for k in k_list]))
        
        max_k = max(k_list)
        top_by_k = compute_top_rmse_by_dim_for_episode(episode_transitions, model, max_k, device, 5)
        
        for idx, _ in top_by_k[1]: top_dim_counts_k1[idx] = top_dim_counts_k1.get(idx, 0) + 1


        print("Top dims k-1 :", " | ".join([f"({idx}):{val:.4f}" for idx, val in top_by_k[1]]))
        print(f"Top dims k-{max_k}:", " | ".join([f"({idx}):{val:.4f}" for idx, val in top_by_k[max_k]]))
        
    print("\n[summary]")
    mean_rmse_by_k = {k: float(np.mean(rmse_values_by_k[k])) for k in k_list}
    print("- RMSE mean:", " | ".join([f"k-{k} {mean_rmse_by_k[k]:.4f}" for k in k_list]))
    
    reward_mean = float(np.mean(episode_rewards))
    reward_std = float(np.std(episode_rewards))  # population std (ddof=0), matches what you wrote
    print(f"- reward: {reward_mean:.4f} ± {reward_std:.4f}")
    
    fp_mean = float(np.mean(episode_forward_progresses))
    fp_std = float(np.std(episode_forward_progresses))  # ddof=0
    print(f"- forward_progress: {fp_mean:.4f} ± {fp_std:.4f}")
    
    elapsed = time.time() - eval_start_time
    elapsed_str = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
    print(f"- elapsed: {elapsed_str}")
    
    print("- top_dims_k1_freq:", end=" ")
    top_dims_sorted = sorted(top_dim_counts_k1.items(), key=lambda kv: kv[1], reverse=True)
    print(" | ".join([f"({idx})x{cnt}" for idx, cnt in top_dims_sorted[:10]]))


    

    
    
    







                
