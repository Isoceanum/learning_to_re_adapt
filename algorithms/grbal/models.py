"""PyTorch modules for the GrBAL dynamics model and adaptation utilities."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.stateless import functional_call


@dataclass
class NormalizationStats:
    """Container for normalization tensors."""

    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    delta_mean: torch.Tensor
    delta_std: torch.Tensor


def build_mlp(input_dim: int, output_dim: int, hidden_sizes: Iterable[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last_dim, size))
        layers.append(nn.ReLU())
        last_dim = size
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class GrBALDynamicsModel(nn.Module):
    """Deterministic MLP dynamics model with support for fast adaptation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (512, 512),
        norm_eps: float = 1e-10,
        predict_delta: bool = True,
        learnable_inner_lr: bool = True,
        inner_lr_init: float = 1e-2,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.predict_delta = bool(predict_delta)
        self._norm_eps = float(norm_eps)
        self.network = build_mlp(self.state_dim + self.action_dim, self.state_dim, hidden_sizes)
        if learnable_inner_lr:
            self.inner_lr = nn.Parameter(torch.tensor(inner_lr_init, dtype=torch.float32))
        else:
            self.register_buffer("inner_lr", torch.tensor(inner_lr_init, dtype=torch.float32))

        self.register_buffer("state_mean", torch.zeros(self.state_dim))
        self.register_buffer("state_std", torch.ones(self.state_dim))
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))
        self.register_buffer("delta_mean", torch.zeros(self.state_dim))
        self.register_buffer("delta_std", torch.ones(self.state_dim))

    # ------------------------------------------------------------------
    # Normalization utilities
    # ------------------------------------------------------------------
    def update_normalization(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> None:
        """Fit normalization statistics on the given batch."""
        device = self.state_mean.device
        states_np = states.detach().to(device)
        actions_np = actions.detach().to(device)
        deltas_np = next_states.detach().to(device) - states_np

        state_mean = states_np.mean(dim=0)
        state_std = states_np.std(dim=0) + self._norm_eps
        action_mean = actions_np.mean(dim=0)
        action_std = actions_np.std(dim=0) + self._norm_eps
        delta_mean = deltas_np.mean(dim=0)
        delta_std = deltas_np.std(dim=0) + self._norm_eps

        self.state_mean.copy_(state_mean)
        self.state_std.copy_(state_std)
        self.action_mean.copy_(action_mean)
        self.action_std.copy_(action_std)
        self.delta_mean.copy_(delta_mean)
        self.delta_std.copy_(delta_std)

    def get_normalization(self) -> NormalizationStats:
        return NormalizationStats(
            state_mean=self.state_mean.clone(),
            state_std=self.state_std.clone(),
            action_mean=self.action_mean.clone(),
            action_std=self.action_std.clone(),
            delta_mean=self.delta_mean.clone(),
            delta_std=self.delta_std.clone(),
        )

    def load_normalization(self, stats: NormalizationStats) -> None:
        self.state_mean.copy_(stats.state_mean)
        self.state_std.copy_(stats.state_std)
        self.action_mean.copy_(stats.action_mean)
        self.action_std.copy_(stats.action_std)
        self.delta_mean.copy_(stats.delta_mean)
        self.delta_std.copy_(stats.delta_std)

    # ------------------------------------------------------------------
    # Forward utilities
    # ------------------------------------------------------------------
    def _prepare_inputs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        norm_states = (states - self.state_mean) / self.state_std
        norm_actions = (actions - self.action_mean) / self.action_std
        return torch.cat([norm_states, norm_actions], dim=-1)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        inputs = self._prepare_inputs(states, actions)
        if params is None:
            delta = self.network(inputs)
        else:
            delta = functional_call(self.network, params, (inputs,))
        if self.predict_delta:
            delta = delta * self.delta_std + self.delta_mean
        return delta

    def predict_next_state(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        delta = self.forward(states, actions, params=params)
        return states + delta if self.predict_delta else delta

    # ------------------------------------------------------------------
    # Losses and adaptation
    # ------------------------------------------------------------------
    def loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        preds = self.predict_next_state(states, actions, params=params)
        return F.mse_loss(preds, next_states)

    def named_parameter_dict(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict((name, param) for name, param in self.network.named_parameters())

    def adapt(
        self,
        params: OrderedDict[str, torch.Tensor],
        loss: torch.Tensor,
        create_graph: bool,
    ) -> OrderedDict[str, torch.Tensor]:
        grads = torch.autograd.grad(
            loss,
            tuple(params.values()),
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )
        step_size = torch.clamp(self.inner_lr, min=1e-6)
        updated = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            updated[name] = param - step_size * grad
        return updated


def detach_params(params: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Create a detached copy of a parameter dict with gradients enabled."""
    cloned = OrderedDict()
    for name, tensor in params.items():
        cloned[name] = tensor.detach().clone().requires_grad_(True)
    return cloned
