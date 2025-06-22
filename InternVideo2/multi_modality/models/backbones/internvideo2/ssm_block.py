import torch
from torch import nn

class SimpleStateSpaceBlock(nn.Module):
    """A lightweight State-Space block for streaming video."""
    def __init__(self, input_size: int, hidden_size: int, gating: bool = True):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.state_proj = nn.Linear(hidden_size, hidden_size)
        self.gating = gating
        if gating:
            self.gate = nn.Linear(input_size, hidden_size)
        self.norm = nn.modules.normalization.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor):
        """Process one step.

        Args:
            x: (B, input_size) current input feature.
            prev_state: (B, hidden_size) previous state.
        Returns:
            new_state: (B, hidden_size)
        """
        inp = self.input_proj(x)
        state_update = self.state_proj(prev_state)
        if self.gating:
            gate = torch.sigmoid(self.gate(x))
            state_update = gate * (state_update + inp)
        else:
            state_update = state_update + inp
        new_state = self.norm(state_update)
        return new_state
