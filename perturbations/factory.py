"""Factory utilities for constructing perturbation instances from config dictionaries."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from .action_scaling_perturbation import ActionScalingPerturbation


def build_perturbation_from_config(config: Optional[Dict[str, Any]]):
    """Return a perturbation instance defined by ``config``.

    Args:
        config: Parsed YAML configuration for a perturbation. ``None`` disables perturbations.

    Returns:
        A concrete ``Perturbation`` subclass instance or ``None`` when no config is provided.
    """
    if not config:
        return None

    cfg = copy.deepcopy(config)
    perturb_type = str(cfg.get("type", "")).strip().lower()

    if perturb_type == "action_scaling":
        return _build_action_scaling(cfg)

    raise ValueError(f"Unknown perturbation type '{perturb_type}' in config: {config}")


def _build_action_scaling(cfg: Dict[str, Any]):
    indices = cfg.get("effected_action_indices")
    if not indices:
        raise ValueError("Action scaling perturbation requires 'effected_action_indices'.")

    bounds = cfg.get("range") or cfg.get("scale_range")
    if not bounds or len(bounds) != 2:
        raise ValueError("Action scaling perturbation requires 'range': [low, high].")

    probability = float(cfg.get("probability", 1.0))
    name = cfg.get("name")

    return ActionScalingPerturbation(
        effected_action_indices=[int(i) for i in indices],
        scale_range=(float(bounds[0]), float(bounds[1])),
        probability=probability,
        name=name,
    )
