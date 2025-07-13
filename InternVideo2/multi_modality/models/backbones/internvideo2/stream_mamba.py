from .internvideo2_clip_vision import CrossAttention, AttentiveBlock, AttentionPoolingBlock, RMSNorm, LayerScale, Attention, Mlp, Block, PatchEmbed

from .mobileclip import TextTransformer, ClipTokenizer, VisionTransformer, vit_b16
from .video_mamba_block import VideoMambaBlock, CrossMambaFiLM, CrossMambaSPFS

import logging
import numpy as np
import torch
import timm
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch import nn

from .pos_embed import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed

# Streaming Student Model
class StreamMamba(nn.Module):
    def __init__(
            self,
            # Parameters for the MobileCLIP ViT
            vit_lite_model_name="vit_b16",
            vit_lite_proj_dim=512, # Projection dimension
            vit_lite_embed_dim=768, # Output dimension
            # RNN parameters
            rnn_type='lstm', # 'lstm', 'gru', or 'mamba'
            rnn_hidden_size=1024,
            rnn_num_layers=1,
            rnn_dropout=0.0, # Dropout for RNN layers (if rnn_num_layers > 1)
            # Output FC layers parameters
            fc_hidden_layers=[512], # List of hidden layer sizes for FC part, empty for direct projection
            teacher_clip_embed_dim=768, # Dimension of the teacher's output
            text_embed_dim=None,
            pred_rank=32,
        ):
        super().__init__()

        # MobileCLIP VisionTransformer class.
        self.vit_lite = timm.create_model(
            vit_lite_model_name,
            projection_dim = vit_lite_proj_dim
        )

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_type = rnn_type.lower()
        self.pred_rank = pred_rank
        self.text_embed_dim = text_embed_dim if text_embed_dim is not None else teacher_clip_embed_dim

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=vit_lite_embed_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                dropout=rnn_dropout if rnn_num_layers > 1 else 0.0
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=vit_lite_embed_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=True,
                dropout=rnn_dropout if rnn_num_layers > 1 else 0.0
            )
        elif self.rnn_type == 'mamba':
            self.rnn = VideoMambaBlock(
                in_dim=vit_lite_embed_dim,
                hidden_dim=rnn_hidden_size,
                clip_dim=teacher_clip_embed_dim,
            )
        elif self.rnn_type == 'cross_mamba_film':
            text_dim = text_embed_dim if text_embed_dim is not None else teacher_clip_embed_dim
            self.rnn = CrossMambaFiLM(
                in_dim=vit_lite_embed_dim,
                hidden_dim=rnn_hidden_size,
                clip_dim=teacher_clip_embed_dim,
                text_dim=text_dim,
            )
        elif self.rnn_type == 'stream_mamba':
            text_dim = text_embed_dim if text_embed_dim is not None else teacher_clip_embed_dim
            self.rnn = CrossMambaSPFS(
                in_dim=vit_lite_embed_dim,
                hidden_dim=rnn_hidden_size,
                clip_dim=teacher_clip_embed_dim,
                text_dim=text_dim,
                pred_rank=pred_rank,
            )
        else:
            raise NotImplementedError(
                f"Unsupported RNN type: {rnn_type}. Choose 'lstm', 'gru', 'mamba', 'cross_mamba_film' or 'stream_mamba'."
            )

        # Fully Connected layers to project RNN output to teacher's embedding dimension
        if self.rnn_type != 'mamba':
            fc_layers = []
            current_dim = rnn_hidden_size
            if fc_hidden_layers:
                for h_dim in fc_hidden_layers:
                    fc_layers.append(nn.Linear(current_dim, h_dim))
                    fc_layers.append(nn.ReLU())
                    current_dim = h_dim
            fc_layers.append(nn.Linear(current_dim, teacher_clip_embed_dim))
            self.output_fc = nn.Sequential(*fc_layers)
        else:
            self.output_fc = nn.Identity()

    def init_hidden(self, batch_size, device):
        if self.rnn_type in ['mamba', 'cross_mamba_film', 'stream_mamba']:
            state = self.rnn.init_state(batch_size, device)
            if self.rnn_type == 'stream_mamba':
                self.rnn.last_hidden = None
            return state
        h0 = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
            return (h0, c0)
        return h0

    def forward(self, single_frame_input, prev_hidden_state, gamma=None, beta=None, tau=None):
        """
        Processes a single frame (or a small chunk of frames) and updates the hidden state.

        Args:
            single_frame_input (torch.Tensor): Input frame(s) for the ViT-Lite.
                Shape: (B, C, H, W) if student_num_frames_processed_by_vit=1
                Shape: (B, C, T_chunk, H, W) if student_num_frames_processed_by_vit > 1
            prev_hidden_state (tuple or torch.Tensor): Previous hidden state from the RNN.
                For LSTM: (h_prev, c_prev)
                For GRU: h_prev

        Returns:
            student_embedding (torch.Tensor): The output embedding for the current step.
                                            Shape: (B, teacher_clip_embed_dim)
            current_hidden_state (tuple or torch.Tensor): The updated RNN hidden state.
        """
        # single_frame_input shape: (B, C, T_chunk, H, W) or (B, C, H, W)
        # ViT-Lite expects (B, C, H, W)

        if len(single_frame_input.shape) == 5:
            single_frame_input = single_frame_input.squeeze(2)  # Remove the T_chunk dimension

        if self.rnn_type == 'stream_mamba':
            threshold = 0.7 if tau is None else tau
            if self.rnn.last_hidden is not None:
                mu, logvar = self.rnn.predict_next_feat()
                conf = torch.exp(-logvar)
                if torch.all(conf > threshold):
                    frame_feature = mu
                else:
                    frame_feature, _ = self.vit_lite.extract_features(single_frame_input)
            else:
                frame_feature, _ = self.vit_lite.extract_features(single_frame_input)
            student_embedding, current_hidden_state = self.rnn(frame_feature, prev_hidden_state, gamma, beta)
            return student_embedding, current_hidden_state

        frame_feature, _ = self.vit_lite.extract_features(single_frame_input)  # (B, student_embed_dim)

        if self.rnn_type == 'mamba':
            student_embedding, current_hidden_state = self.rnn(frame_feature, prev_hidden_state)
            return student_embedding, current_hidden_state
        elif self.rnn_type == 'cross_mamba_film':
            student_embedding, current_hidden_state = self.rnn(frame_feature, prev_hidden_state, gamma, beta)
            return student_embedding, current_hidden_state
        student_embedding = self.output_fc(rnn_output_last_step)

        return student_embedding, current_hidden_state

if __name__ == '__main__':
    # Configuration for the student model
    batch_size = 2
    img_size = 224 # Should match teacher's patch processing
    patch_size = 14
    teacher_output_dim = 768 # Example: output dimension of full InternVideo2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_config = {
        "vit_lite_model_name": "vit_b16",
        "vit_lite_proj_dim": 512,
        "vit_lite_embed_dim": 768,
        "rnn_type": 'mamba',
        "rnn_hidden_size": 512,
        "rnn_num_layers": 1,
        "rnn_dropout": 0.0,
        "fc_hidden_layers": [256],
        "teacher_clip_embed_dim": teacher_output_dim,
    }

    student_model = StreamMamba(**student_config).to(device)
    student_model.eval()

    print(f"Student model created with {sum(p.numel() for p in student_model.parameters())/1e6:.2f}M parameters.")

    # Simulate streaming a few frames
    num_stream_steps = 5
    current_hidden = student_model.init_hidden(batch_size, device)
    gamma = beta = None
    if student_config["rnn_type"] == 'cross_mamba_film':
        dummy_prompt = torch.randn(1, teacher_output_dim).to(device)
        gamma, beta = student_model.rnn.prepare_prompt(dummy_prompt)

    for i in range(num_stream_steps):
        # Create a dummy single frame input for each step
        # For ViT-Lite processing 1 frame: (B, C, H, W)
        dummy_frame = torch.randn(batch_size, 3, img_size, img_size).to(device)

        with torch.no_grad():
            output_embedding, current_hidden = student_model(dummy_frame, current_hidden, gamma, beta)

        print(f"Step {i+1}: Output embedding shape: {output_embedding.shape}")
        if student_config["rnn_type"] == 'lstm':
            print(f"  LSTM hidden state h shape: {current_hidden[0].shape}, c shape: {current_hidden[1].shape}")
        elif student_config["rnn_type"] == 'gru':
            print(f"  GRU hidden state shape: {current_hidden.shape}")
        else:
            conv_state, ssm_state = current_hidden
            print(f"  Mamba conv state: {conv_state.shape}, ssm state: {ssm_state.shape}")
