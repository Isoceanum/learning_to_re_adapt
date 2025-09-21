"""Buffers for meta-learning sequence sampling in GrBAL."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class MetaBatch:
    context_states: torch.Tensor
    context_actions: torch.Tensor
    context_next_states: torch.Tensor
    target_states: torch.Tensor
    target_actions: torch.Tensor
    target_next_states: torch.Tensor


class MetaSequenceBuffer:
    """Stores trajectories and yields contiguous segments for meta-updates."""

    def __init__(self) -> None:
        self._paths: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add_path(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray) -> None:
        assert states.shape == next_states.shape
        assert states.shape[0] == actions.shape[0]
        self._paths.append((states.copy(), actions.copy(), next_states.copy()))

    def __len__(self) -> int:
        return sum(path[0].shape[0] for path in self._paths)

    def sample(
        self,
        batch_size: int,
        context_len: int,
        target_len: int,
        device: torch.device,
    ) -> MetaBatch:
        if not self._paths:
            raise RuntimeError("MetaSequenceBuffer is empty")
        total_len = context_len + target_len
        ctx_states, ctx_actions, ctx_next_states = [], [], []
        tgt_states, tgt_actions, tgt_next_states = [], [], []

        trials = 0
        while len(ctx_states) < batch_size:
            trials += 1
            if trials > batch_size * 20:
                raise RuntimeError("Insufficient path length to sample the requested meta-batch")
            states, actions, next_states = random.choice(self._paths)
            if states.shape[0] < total_len:
                continue
            max_start = states.shape[0] - total_len
            idx = random.randint(0, max_start) if max_start > 0 else 0
            ctx_slice = slice(idx, idx + context_len)
            tgt_slice = slice(idx + context_len, idx + total_len)
            ctx_states.append(states[ctx_slice])
            ctx_actions.append(actions[ctx_slice])
            ctx_next_states.append(next_states[ctx_slice])
            tgt_states.append(states[tgt_slice])
            tgt_actions.append(actions[tgt_slice])
            tgt_next_states.append(next_states[tgt_slice])

        def to_tensor(arrays: List[np.ndarray]) -> torch.Tensor:
            return torch.tensor(np.stack(arrays, axis=0), dtype=torch.float32, device=device)

        return MetaBatch(
            context_states=to_tensor(ctx_states),
            context_actions=to_tensor(ctx_actions),
            context_next_states=to_tensor(ctx_next_states),
            target_states=to_tensor(tgt_states),
            target_actions=to_tensor(tgt_actions),
            target_next_states=to_tensor(tgt_next_states),
        )
