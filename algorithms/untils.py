
import time

import torch
from algorithms.common.dynamics_model import DynamicsModel
from algorithms.common.planner import RandomShootingPlanner, CrossEntropyMethodPlanner, MPPIPlanner

def make_dynamics_model(dynamics_model_config, obs_dim, action_dim, seed):
    if dynamics_model_config is None: 
        raise AttributeError("Missing dynamics_model config in YAML")
    
    hidden_sizes = dynamics_model_config["train"]["dynamics_model"]["hidden_sizes"]
    learning_rate = float(dynamics_model_config["train"]["dynamics_model"]["learning_rate"])

    return DynamicsModel(obs_dim, action_dim, hidden_sizes, learning_rate, seed)


def make_planner(planner_config, dynamics_fn, reward_fn, action_space, device, seed):
    if planner_config is None:
        raise AttributeError("Missing planner config in YAML")
    
    planner_type = planner_config.get("type")         
    horizon = int(planner_config.get("horizon"))
    n_candidates = int(planner_config.get("n_candidates"))
    discount = float(planner_config.get("discount"))
    
    act_low = action_space.low
    act_high = action_space.high
            
    if planner_type == "rs":
        return RandomShootingPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount)
    
    if planner_type == "cem":
        num_cem_iters = int(planner_config.get("num_cem_iters"))
        percent_elites = float(planner_config.get("percent_elites"))
        alpha = float(planner_config.get("alpha"))        
        return CrossEntropyMethodPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount, num_cem_iters, percent_elites, alpha, seed)
    
    if planner_type == "mppi":
        noise_sigma = float(planner_config.get("noise_sigma"))
        lambda_ = float(planner_config.get("lambda_"))
        return MPPIPlanner(dynamics_fn, reward_fn, horizon, n_candidates, act_low, act_high, device, discount, noise_sigma, lambda_, seed)
        
    raise AttributeError(f"Planner type {planner_type} not supported")



def collect_env_steps(env, policy_fn, step_env_fn, storage, step_target, max_path_length):
    steps_collected_this_iteration = 0
        
    log_collect_start_time = time.time()
    log_episodes = 0
    log_episode_forward_progress = []
    log_episode_velocity = []
    log_episode_returns = []
    
    while steps_collected_this_iteration < step_target:
        obs, _ = env.reset()
        log_episodes += 1
        
        episode_return = 0.0
        episode_x_start = None
        episode_x_last = None
        episode_velocity = 0.0

        episode_steps = 0          
        episode_obs = []
        episode_act = []
        episode_next_obs = []
        
        while episode_steps < max_path_length:
            
            action = policy_fn(obs)
            if torch.is_tensor(action):
                action = action.detach().cpu().numpy()
                
            next_obs, reward, terminated, truncated, info = step_env_fn(action)
            episode_return += float(reward)
                    
            x_position = float(info["x_position"])
            if episode_x_start is None:
                episode_x_start = x_position
            episode_x_last = x_position
            
            episode_velocity += float(info["x_velocity"])
            
            episode_obs.append(obs)
            episode_act.append(action)
            episode_next_obs.append(next_obs)
            
            obs = next_obs
        
            episode_steps += 1
            steps_collected_this_iteration += 1
                        
            if steps_collected_this_iteration >= step_target:
                break
            
            if terminated or truncated:
                break
            
        storage.add_trajectory(episode_obs, episode_act, episode_next_obs)
        log_episode_forward_progress.append(float(episode_x_last - episode_x_start))
        log_episode_velocity.append(float(episode_velocity))
        log_episode_returns.append(float(episode_return))
                
    num_train_transitions = sum(len(ep) for ep in storage.train_observations)
    avg_reward = sum(log_episode_returns) / max(1, len(log_episode_returns))
    avg_forward_progress = sum(log_episode_forward_progress) / max(1, len(log_episode_forward_progress))
    avg_velocity = sum(log_episode_velocity) / max(1, len(log_episode_velocity))
    steps_collected_this_iteration = steps_collected_this_iteration
    log_collect_time = time.time() - log_collect_start_time, 
    log_episodes = log_episodes
    
    print(f"collect: dataset={num_train_transitions} " f"steps={steps_collected_this_iteration} " f"episodes={log_episodes} " f"avg_rew={avg_reward:.3f} " f"avg_fp={avg_forward_progress:.3f} " f"avg_v={avg_velocity:.3f} " f"time={log_collect_time:.1f}s")
