import numpy as np
from typing import Any, Dict, Optional

import gymnasium as gym


class PerturbationWrapper(gym.Wrapper):
    """
    Minimal per-episode perturbation wrapper.

    Config (perturb_cfg):
      - type: currently supports "cripple"
      - prob: probability in [0,1] that perturbation is active for the episode
      - target: list of action indices eligible for perturbation

    Semantics (cripple): if active, zeros the chosen action dimension every step.
    """

    def __init__(self, env: gym.Env, perturb_cfg: Optional[Dict[str, Any]] = None):
        super().__init__(env)
        cfg = perturb_cfg or {}
        self._ptype: str = str(cfg.get("type", "cripple")).lower()
        self._prob: float = float(cfg.get("prob", 0.0))
        targets = cfg.get("target", [])
        if isinstance(targets, (list, tuple)):
            self._targets = [int(t) for t in targets]
        elif targets is None:
            self._targets = []
        else:
            # single int
            try:
                self._targets = [int(targets)]
            except Exception:
                self._targets = []

        # Episode state
        self._active: bool = False
        self._target_idx: Optional[int] = None

    # Public accessor for debugging
    @property
    def current_perturbation(self) -> Dict[str, Any]:
        return {
            "active": self._active,
            "type": self._ptype,
            "target": self._target_idx,
        }

    def reset(self, *args, **kwargs):
        self._sample_episode_perturbation()
        out = self.env.reset(*args, **kwargs)
        # Attach info field if Gymnasium API returns (obs, info)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            obs, info = out
            info = dict(info)
            info["perturbation"] = self.current_perturbation
            return obs, info
        return out

    def step(self, action):
        a = action
        if self._ptype == "cripple" and self._active and self._target_idx is not None:
            try:
                # Create a copy if the incoming action might be reused upstream
                a = np.array(action, copy=True)
                if 0 <= self._target_idx < a.shape[-1]:
                    a[self._target_idx] = 0.0
            except Exception:
                # Keep it simple: on any issue, fall back to original action
                a = action

        out = self.env.step(a)

        # Insert perturbation info for both Gymnasium 5-tuple and legacy 4-tuple
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            info = dict(info) if isinstance(info, dict) else {}
            info["perturbation"] = self.current_perturbation
            return obs, reward, terminated, truncated, info
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            info = dict(info) if isinstance(info, dict) else {}
            info["perturbation"] = self.current_perturbation
            return obs, reward, done, info
        else:
            return out

    # --- helpers ---
    def _sample_episode_perturbation(self):
        self._active = False
        self._target_idx = None

        if self._ptype != "cripple":
            return
        if self._prob <= 0.0:
            return
        if not self._targets:
            return

        # Activate with probability prob and pick a target index uniformly
        if float(np.random.rand()) < float(self._prob):
            target = int(np.random.choice(self._targets))
            # Basic bounds check w.r.t. action space
            try:
                dim = int(self.action_space.shape[0])
                if 0 <= target < dim:
                    self._active = True
                    self._target_idx = target
            except Exception:
                # If action space shape is unavailable, still activate
                self._active = True
                self._target_idx = target

# Backward compatibility alias
SimplePerturbationWrapper = PerturbationWrapper

