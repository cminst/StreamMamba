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
        out, conv_state, ssm_state = self.ssm.step(gated.unsqueeze(1), conv_state, ssm_state, A_scale=None)
        out = out.squeeze(1)
        out = out + gated
        out = out * torch.sigmoid(self.out_gate(out))
        self._hidden = out
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

    def forward(self, frame_feat, state, gamma=None, beta=None, tau=None):
        if gamma is not None and beta is not None:
            frame_feat = gamma * frame_feat + beta
        return super().forward(frame_feat, state)


class CrossMambaSPFS(CrossMambaFiLM):
    """CrossMambaFiLM with a low-rank feature predictor for SPFS."""

    def __init__(self, *args, pred_rank=32, **kw):
        super().__init__(*args, **kw)
        d = self.proj.out_features
        r = pred_rank
        self.pred_U = nn.Parameter(torch.randn(d, r))
        self.pred_V = nn.Linear(self.ssm.d_model, r, bias=False)
        self.logvar = nn.Linear(self.ssm.d_model, 1)
        self.last_hidden = None

    def forward(self, frame_feat, state, gamma=None, beta=None, tau=None):
        clip_emb, state = super().forward(frame_feat, state, gamma, beta, tau)
        # Store hidden representation for next-step prediction
        self.last_hidden = self._hidden
        return clip_emb, state

    def predict_next_feat(self, h=None):
        if h is None:
            if self.last_hidden is None:
                raise RuntimeError("No hidden state available for prediction")
            h = self.last_hidden
        mu = self.pred_V(h) @ self.pred_U.T
        unc = self.logvar(h).squeeze(-1)
        return mu, unc
