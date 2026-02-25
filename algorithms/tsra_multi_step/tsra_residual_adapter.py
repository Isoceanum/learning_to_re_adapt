import torch
import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
}


class TSRAResidualAdapter(nn.Module):
    """Configurable residual adapter for TSRA K-step training."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        input_mode: str,
        hidden_sizes,
        activation: str = "relu",
        output_mode: str = "delta_next_obs",
        zero_init_last_layer: bool = True,
        bound_type: str = "none",
        bound_scale: float = 1.0,
        bound_max_abs: float = 0.0,
    ):
        super().__init__()

        if input_mode not in ("sa", "sabase"):
            raise ValueError(f"input_mode must be 'sa' or 'sabase', got {input_mode}")
        if output_mode != "delta_next_obs":
            raise ValueError(f"output_mode supports only 'delta_next_obs', got {output_mode}")

        if activation.lower() not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(_ACTIVATIONS.keys())}")
        act_layer = _ACTIVATIONS[activation.lower()]

        in_dim = observation_dim + action_dim
        if input_mode == "sabase":
            in_dim += observation_dim

        layers = []
        last_dim = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(act_layer())
            last_dim = h

        final = nn.Linear(last_dim, observation_dim)
        if zero_init_last_layer:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        layers.append(final)

        self.model = nn.Sequential(*layers)
        self.input_mode = input_mode
        self.bound_type = bound_type
        self.bound_scale = float(bound_scale)
        self.bound_max_abs = float(bound_max_abs)

    def forward(self, obs_norm, act_norm, base_pred_next_obs_norm=None):
        if self.input_mode == "sabase":
            if base_pred_next_obs_norm is None:
                raise ValueError("base_pred_next_obs_norm required for input_mode='sabase'")
            x = torch.cat([obs_norm, act_norm, base_pred_next_obs_norm], dim=-1)
        else:
            x = torch.cat([obs_norm, act_norm], dim=-1)

        delta = self.model(x)

        if self.bound_type == "tanh":
            delta = torch.tanh(delta) * self.bound_scale
        elif self.bound_type == "clamp" and self.bound_max_abs > 0:
            delta = torch.clamp(delta, -self.bound_max_abs, self.bound_max_abs)

        return delta

    def bound_config(self):
        return {
            "type": self.bound_type,
            "scale": self.bound_scale,
            "max_abs": self.bound_max_abs,
        }

