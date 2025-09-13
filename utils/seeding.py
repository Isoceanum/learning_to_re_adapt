"""
Global seeding helpers for reproducibility.

Added for Nagabandi fidelity
"""

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = False):
    """Seed Python, NumPy, and PyTorch RNGs.

    Added for Nagabandi fidelity
    """
    try:
        import torch
    except Exception:
        torch = None

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Determinism toggles
        try:
            import torch.backends.cudnn as cudnn

            cudnn.benchmark = False
            cudnn.deterministic = bool(deterministic_torch)
        except Exception:
            pass
        # On newer PyTorch versions, optionally enforce determinism
        try:
            torch.use_deterministic_algorithms(bool(deterministic_torch))
        except Exception:
            pass

    # Some libs read this for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_env(env, seed: Optional[int] = None):
    """Seed a Gymnasium env or SB3 VecEnv if possible.

    Added for Nagabandi fidelity
    """
    if seed is None:
        return
    try:
        # Gymnasium single env
        if hasattr(env, "reset"):
            try:
                env.reset(seed=int(seed))
                return
            except TypeError:
                pass
    except Exception:
        pass

    # SB3 VecEnv
    try:
        if hasattr(env, "seed"):
            env.seed(int(seed))  # some VecEnv implementations
    except Exception:
        pass

