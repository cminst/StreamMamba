import math
import torch
from torch import nn
from mamba_ssm.modules.mamba2 import Mamba2
import torch.nn.functional as F

class VideoMambaBlock(nn.Module):
    """State-space block for streaming video features using Mamba.

    The block exposes an interface compatible with the previous LSTM:
        output, new_state = block(frame_feature, prev_state)

    Parameters
    ----------
    in_dim : int
        Dimension of the per-frame input feature from MobileCLIP.
    hidden_dim : int
        Internal hidden size of the Mamba block.
    clip_dim : int
        Dimension of the projected output embedding (usually CLIP size).
    num_heads : int, optional
        Number of channel groups (heads) for the SSM.
    d_state : int, optional
        Expansion factor for the SSM state.
    d_conv : int, optional
        Convolution kernel width.
    """

    def __init__(self, in_dim, hidden_dim, clip_dim, num_heads=4, d_state=64, d_conv=4):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.pre_norm = nn.LayerNorm(in_dim)
        self.in_gate = nn.Linear(in_dim, hidden_dim)
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.ssm = Mamba2(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            ngroups=num_heads,
            layer_idx=0,
        )

        self.out_gate = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, clip_dim)

    def init_state(self, batch_size, device):
        conv, ssm = self.ssm.allocate_inference_cache(batch_size, 1)
        return conv.to(device), ssm.to(device)

    def forward(self, frame_feat, state):
        conv_state, ssm_state = state
        x = self.pre_norm(frame_feat)
        gated = self.input_proj(x) * torch.sigmoid(self.in_gate(x))
        out, conv_state, ssm_state = self.ssm.step(gated.unsqueeze(1), conv_state, ssm_state)
        out = out.squeeze(1)
        out = out + gated
        out = out * torch.sigmoid(self.out_gate(out))
        clip_emb = self.proj(out)
        return clip_emb, (conv_state, ssm_state)


class CrossMambaFiLM(VideoMambaBlock):
    """VideoMambaBlock with FiLM-style text conditioning."""

    def __init__(self, in_dim, hidden_dim, clip_dim, num_heads=4, d_state=64, d_conv=4, text_dim=None):
        super().__init__(in_dim, hidden_dim, clip_dim, num_heads, d_state, d_conv)
        if text_dim is None:
            text_dim = clip_dim
        self.film = nn.Linear(text_dim, 2 * in_dim, bias=True)

    @torch.no_grad()
    def prepare_prompt(self, prompt_vec):
        """Prepare FiLM parameters from text embedding."""
        gamma, beta = self.film(prompt_vec).chunk(2, dim=-1)
        return gamma.sigmoid(), beta

    def forward(self, frame_feat, state, gamma=None, beta=None):
        if gamma is not None and beta is not None:
            frame_feat = gamma * frame_feat + beta
        return super().forward(frame_feat, state)


class TauMamba(VideoMambaBlock):
    """VideoMambaBlock with time-constant scaling based on text embedding."""

    def __init__(self, in_dim, hidden_dim, clip_dim, num_heads=4, d_state=64, d_conv=4, text_dim=None):
        super().__init__(in_dim, hidden_dim, clip_dim, num_heads, d_state, d_conv)
        if text_dim is None:
            text_dim = clip_dim
        self.tau_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        # Cache the initial A parameter for scaling
        with torch.no_grad():
            self.register_buffer("A_base", (-torch.exp(self.ssm.A_log)).clone())

    @torch.no_grad()
    def prepare_prompt(self, prompt_vec):
        """Compute per-layer tau from text embedding."""
        return self.tau_mlp(prompt_vec).sigmoid().clamp(0.05, 0.995)

    def forward(self, frame_feat, state, tau=None):
        conv_state, ssm_state = state
        if tau is not None:
            # Apply tau by scaling the transition matrix A once per step
            self.ssm.A_scale = tau.squeeze()
        else:
            self.ssm.A_scale = None
        x = self.pre_norm(frame_feat)
        gated = self.input_proj(x) * torch.sigmoid(self.in_gate(x))
        out, conv_state, ssm_state = self.ssm.step(gated.unsqueeze(1), conv_state, ssm_state)
        out = out.squeeze(1)
        out = out + gated
        out = out * torch.sigmoid(self.out_gate(out))
        clip_emb = self.proj(out)
        return clip_emb, (conv_state, ssm_state)


class TauMambaFiLM(CrossMambaFiLM):
    """CrossMambaFiLM with additional tau scaling."""

    def __init__(
        self,
        in_dim,
        hidden_dim,
        clip_dim,
        num_heads=4,
        d_state=64,
        d_conv=4,
        text_dim=None,
        tau_min = 0.05,
        tau_max = 0.995
    ):
        super().__init__(in_dim, hidden_dim, clip_dim, num_heads, d_state, d_conv, text_dim)
        if text_dim is None:
            text_dim = clip_dim

        self.tau_min = tau_min
        self.tau_max = tau_max

        self.tau_mlp = nn.Sequential(
            nn.LayerNorm(text_dim), # stabilise text scale
            nn.Linear(text_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        with torch.no_grad():
            self.register_buffer("A_base", (-torch.exp(self.ssm.A_log)).clone())

        # initialize to output constant 0.9
        for m in self.tau_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                nn.init.zeros_(m.bias)

        tau0 = 0.9
        logit0 = math.log((tau0 - tau_min) / (tau_max - tau0))
        self.tau_mlp[-1].bias.data.fill_(logit0)

    @torch.no_grad()
    def prepare_prompt(self, prompt_vec):
        gamma, beta = super().prepare_prompt(prompt_vec)
        tau = self.tau_mlp(prompt_vec).sigmoid().clamp(0.05, 0.995)
        return gamma, beta, tau

    def forward(self, frame_feat, state, gamma=None, beta=None, tau=None):
        if gamma is not None and beta is not None:
            frame_feat = gamma * frame_feat + beta
        conv_state, ssm_state = state
        if tau is not None:
            self.ssm.A_scale = tau.squeeze()
        else:
            self.ssm.A_scale = None
        x = self.pre_norm(frame_feat)
        gated = self.input_proj(x) * torch.sigmoid(self.in_gate(x))
        out, conv_state, ssm_state = self.ssm.step(gated.unsqueeze(1), conv_state, ssm_state)
        out = out.squeeze(1)
        out = out + gated
        out = out * torch.sigmoid(self.out_gate(out))
        clip_emb = self.proj(out)
        return clip_emb, (conv_state, ssm_state)
