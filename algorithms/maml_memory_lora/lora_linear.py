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
        self.scale = self.alpha / self.r if self.r > 0 else 0.0

        self.A = nn.Linear(in_features, self.r, bias=False) if self.r > 0 else None
        self.B = nn.Linear(self.r, out_features, bias=False) if self.r > 0 else None

        if base_linear is not None and self.r > 0:
            base_weight = base_linear.weight
            self.A = self.A.to(device=base_weight.device, dtype=base_weight.dtype)
            self.B = self.B.to(device=base_weight.device, dtype=base_weight.dtype)
        
        if self.A is not None: nn.init.normal_(self.A.weight, mean=0.0, std=0.01) # small values like lora paper
        if self.B is not None: nn.init.zeros_(self.B.weight) # zeros like lora paper

    def forward(self, x):
        """
        Compute y = base(x) + scale * (B(A(x))).

        TS-LoRA context:
        - The base path represents the pretrained dynamics model.
        - The LoRA path is the only trainable component for task-specific adaptation.
        """
        
        base_out = self.base(x)
        if self.r == 0: return base_out
        lora_out = self.B(self.A(x))
        return base_out + self.scale * lora_out
        

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
        params = []
        if self.A is not None: params.extend(self.A.parameters())
        if self.B is not None: params.extend(self.B.parameters())
        return params


    def get_lora_state_dict(self):
        """
        Return a dict containing only LoRA tensors (no base weights).

        TS-LoRA context:
        - Enables saving task-specific adapters without duplicating the base model.
        """
        state = {}
        if self.A is not None: state["A"] = self.A.weight.detach().clone()
        if self.B is not None: state["B"] = self.B.weight.detach().clone()
        return state

    def load_lora_state_dict(self, state):
        """
        Load LoRA tensors (A,B) from a provided state dict.

        TS-LoRA context:
        - Enables retrieving task-specific adapters from memory or disk.
        """
        if self.A is not None and "A" in state: self.A.weight.data.copy_(state["A"])
        if self.B is not None and "B" in state: self.B.weight.data.copy_(state["B"])
