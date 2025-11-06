
import random
import numpy as np
import torch

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def seed_env(env, seed):
    seed = int(seed)
    if hasattr(env, "reset"):
        env.reset(seed=seed)
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
