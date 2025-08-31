from scripts.train_hf_ppo_ant import train_hf_ppo_ant
from scripts.eval_hf_ppo_ant import eval_hf_ppo_ant
from scripts.smoke_test_half_cheetah import smoke_test_half_cheetah

from scripts.train_half_cheetah import train_half_cheetah

from scripts.eval_half_cheetah import eval_hf_ppo_half_cheetah


import time


if __name__ == "__main__":
    print("starting main")
    
    start_time = time.time()  # start timer
    #train_hf_ppo_ant(total_timesteps=2_000_000)

    
    # Run a quick HalfCheetah smoke test with random actions and rendering
    #smoke_test_half_cheetah(steps=200, seed=42, render=True)

    # You can also evaluate the Ant PPO model (if trained):
    # eval_hf_ppo_ant(episodes=3, render=True)
    
    #train_half_cheetah(total_timesteps=8_000_000, seed=42, n_envs=8)
    elapsed = time.time() - start_time
    print(f"⏱️ Training took {elapsed/60:.2f} minutes")
    
    eval_hf_ppo_half_cheetah(episodes=100, render=False)

