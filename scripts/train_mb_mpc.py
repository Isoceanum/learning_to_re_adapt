#!/usr/bin/env python3
"""
Train MB-MPC (Model-Based MPC with learned dynamics) on Mujoco envs.
- Default env: HopperCustom-v0
- Uses CEMPlanner by default.
- Saves final dynamics model to outputs/mb_mpc_dynamics_hopper.pt (by default)
"""

import gymnasium as gym
from algorithms.mb_mpc.trainer import DynamicsTrainer

    
def train_mb_mpc(env_name="HopperCustom-v0",
                 total_iterations=10,
                 init_random_steps=2000,      # 2–3 episodes of random play
                 rollout_steps=5000,          # more data per iteration
                 epochs=30,                   # dynamics model trains more fully
                 horizon=20,                  # longer planning horizon
                 num_candidates=1000,         # more candidates for better plans
                 device="cpu",
                 save_path="outputs/mb_mpc_dynamics_hopper.pt"):

    # Load env with x-position included for correct x-velocity in planner
    env = gym.make(env_name, exclude_current_positions_from_observation=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Pull environment control-cost coefficient if available
    try:
        ctrl_cost_weight = float(getattr(env.unwrapped, "_ctrl_cost_weight"))
    except Exception:
        ctrl_cost_weight = 0.1

    # Initialize trainer
    trainer = DynamicsTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=env.action_space,
        horizon=horizon,
        num_candidates=num_candidates,
        device=device,
        ctrl_cost_weight=ctrl_cost_weight,
    )

    # Run training loop
    trainer.run_training_loop(
        env,
        n_iterations=total_iterations,
        init_random_steps=init_random_steps,
        rollout_steps=rollout_steps,
        epochs=epochs,
        save_path=save_path
    )

    env.close()


if __name__ == "__main__":
    train_mb_mpc()
