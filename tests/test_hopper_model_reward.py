import random
import unittest

import gymnasium as gym
import numpy as np
import torch
import envs  # noqa: F401



class TestHopperModelReward(unittest.TestCase):
    def test_model_reward_matches(self):
        seed = 1234
        env = gym.make(
            "AntCustom-v0",
            exclude_current_positions_from_observation=False,
        )
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env.action_space.seed(seed)

            reward_fn = env.unwrapped.get_model_reward_fn()
            obs, _ = env.reset(seed=seed)

            real_rewards = []
            model_rewards = []
            sample_pairs = []
            target_samples = 20000

            while len(real_rewards) < target_samples:
                action = env.action_space.sample()
                next_obs, real_reward, terminated, truncated, _ = env.step(action)

                state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
                next_state_tensor = torch.tensor(
                    next_obs, dtype=torch.float32
                ).unsqueeze(0)

                model_reward = reward_fn(state_tensor, action_tensor, next_state_tensor)
                model_reward_value = model_reward.squeeze().item()

                real_rewards.append(real_reward)
                model_rewards.append(model_reward_value)

                if len(sample_pairs) < 5:
                    sample_pairs.append((real_reward, model_reward_value))

                obs = next_obs
                if terminated or truncated:
                    obs, _ = env.reset()

            avg_abs_diff = float(
                np.mean(np.abs(np.array(real_rewards) - np.array(model_rewards)))
            )
            
            

            print("Sample (real_reward, model_reward) pairs:")
            
            for real_reward, model_reward in sample_pairs:
                print(f"  ({real_reward:.4f}, {model_reward:.4f})")
                  
            print(f"Average absolute difference {avg_abs_diff:.4f}")
            

            self.assertLess(
                avg_abs_diff,
                0.005,
                msg=f"Average absolute difference {avg_abs_diff:.4f} exceeds tolerance",
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
