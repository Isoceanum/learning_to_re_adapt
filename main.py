from scripts.train_hf_ppo_ant import train_hf_ppo_ant
from scripts.eval_hf_ppo_ant import eval_hf_ppo_ant
from scripts.smoke_test_half_cheetah import smoke_test_half_cheetah

from scripts.train_half_cheetah import train_half_cheetah
from scripts.eval_half_cheetah import eval_hf_ppo_half_cheetah

from scripts.train_hopper import train_hopper
from scripts.eval_hopper import eval_hopper

from scripts.train_mb_mpc import train_mb_mpc
from scripts.eval_mb_mpc import eval_mb_mpc

import time

def ppo_hopper():
    # Example: Train PPO on Hopper and then evaluate
    print("starting main")
    start_time = time.time()
    
    #train_hopper(total_timesteps=2_000_000, seed=0, n_envs=8)
    eval_hopper(episodes=3, render=True)

    elapsed = time.time() - start_time
    print(f"⏱️ Run took {elapsed/60:.2f} minutes")
    
    
def mb_mpc_hopper():
    # Example: Train PPO on Hopper and then evaluate
    print("starting main")
    start_time = time.time()
    
    #train_mb_mpc()
    eval_mb_mpc(render=False, episodes=10)

    elapsed = time.time() - start_time
    print(f"⏱️ Run took {elapsed/60:.2f} minutes")
    
    

if __name__ == "__main__":    
    #train_hf_ppo_ant(total_timesteps=2_000_000)

    
    # Run a quick HalfCheetah smoke test with random actions and rendering
    #smoke_test_half_cheetah(steps=200, seed=42, render=True)

    # You can also evaluate the Ant PPO model (if trained):
    # eval_hf_ppo_ant(episodes=3, render=True)
    
    #train_half_cheetah(total_timesteps=8_000_000, seed=42, n_envs=8)
    
    # Run MB-MPC end-to-end smoke test
    #elapsed = time.time() - start_time
    #print(f"⏱️ Training took {elapsed/60:.2f} minutes")
    #eval_hf_ppo_half_cheetah(episodes=100, render=False)
    
    mb_mpc_hopper()
    
    