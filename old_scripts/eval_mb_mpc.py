import torch
import gymnasium as gym
from algorithms.mb_mpc.dynamics import DynamicsModel
from algorithms.mb_mpc.planner import CEMPlanner  # or RandomShootingPlanner

def eval_mb_mpc(env_name="HopperCustom-v0",
                model_path="outputs/mb_mpc_dynamics_hopper.pt",
                render=True,
                episodes=3):
    # Load environment (include x-position for x-velocity reward)
    env = gym.make(
        env_name,
        render_mode="human" if render else None,
        exclude_current_positions_from_observation=False,
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Load trained dynamics model
    model = DynamicsModel(state_dim, action_dim, hidden_sizes=[256,256])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create planner with trained dynamics model and env-specific ctrl penalty
    try:
        ctrl_cost_weight = float(getattr(env.unwrapped, "_ctrl_cost_weight"))
    except Exception:
        ctrl_cost_weight = 0.1
    planner = CEMPlanner(
        model,
        env.action_space,
        horizon=20,
        num_candidates=1000,
        ctrl_cost_weight=ctrl_cost_weight,
    )

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 1000:
            action = planner.plan(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if render:
                env.render()
            done = terminated or truncated

        print(f"Episode {ep+1}: total_reward={total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    eval_mb_mpc()
