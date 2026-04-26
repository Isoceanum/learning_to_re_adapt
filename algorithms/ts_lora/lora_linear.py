import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r, alpha=1.0, bias=True, base_linear=None):
        """
        Create a LoRA-wrapped Linear layer.

        - If base_linear is provided, it is wrapped directly (pretrained weights).
        - Otherwise, a new nn.Linear is created and should later be loaded from
          a pretrained checkpoint by the trainer.

        TS-LoRA context:
        - The trainer will load a pretrained base model and freeze it.
        - This constructor should not freeze by itself; it only sets structure.
        """
        super().__init__()
        self.base = base_linear if base_linear is not None else nn.Linear(in_features, out_features, bias=bias)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r if self.r > 0 else 0.0
        self.merged = False
        self.merge_weights = True

        if self.r > 0:
            self.A = nn.Parameter(self.base.weight.new_zeros((self.r, in_features)))
            self.B = nn.Parameter(self.base.weight.new_zeros((out_features, self.r)))
            nn.init.normal_(self.A, mean=0.0, std=0.01)
            nn.init.zeros_(self.B)
        else:
            self.A = None
            self.B = None

    def _delta_weight(self):
        if self.r <= 0:
            return None
        return (self.B @ self.A) * self.scaling

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.merge_weights or self.r <= 0:
            return self

        if mode:
            if self.merged:
                self.base.weight.data -= self._delta_weight()
                self.merged = False
        else:
            if not self.merged:
                self.base.weight.data += self._delta_weight()
                self.merged = True
        return self

    def forward(self, x):
        """
        Compute y = base(x) + scale * (B(A(x))).

        TS-LoRA context:
        - The base path represents the pretrained dynamics model.
        - The LoRA path is the only trainable component for task-specific adaptation.
        """
        base_out = self.base(x)
        if self.r <= 0 or self.merged:
            return base_out
        lora_out = (x @ self.A.transpose(0, 1) @ self.B.transpose(0, 1)) * self.scaling
        return base_out + lora_out
        

    def freeze_base(self):
        """
        Freeze base weights and bias so only LoRA params train.

        TS-LoRA context:
        - This is called by the trainer after loading the pretrained checkpoint.
        """
        for param in self.base.parameters():
            param.requires_grad = False

    def lora_parameters(self):
        """
        Return an iterator over only LoRA parameters (A and B).

        TS-LoRA context:
        - The trainer builds the optimizer on these params only.
        """
        if self.A is None or self.B is None:
            return []
        return [self.A, self.B]


    def get_lora_state_dict(self):
        """
        Return a dict containing only LoRA tensors (no base weights).

        TS-LoRA context:
        - Enables saving task-specific adapters without duplicating the base model.
        """
        state = {}
        if self.A is not None: state["A"] = self.A.detach().clone()
        if self.B is not None: state["B"] = self.B.detach().clone()
        return state

    def load_lora_state_dict(self, state):
        """
        Load LoRA tensors (A,B) from a provided state dict.

        TS-LoRA context:
        - Enables retrieving task-specific adapters from memory or disk.
        """
        if self.A is not None and "A" in state: self.A.data.copy_(state["A"])
        if self.B is not None and "B" in state: self.B.data.copy_(state["B"])
