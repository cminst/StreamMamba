from .video_mamba_block import VideoMambaBlock, CrossMambaFiLM, MambaSPFS
from easydict import EasyDict as edict

import torch
import timm
from torch import nn

# Streaming Student Model
class StreamMamba(nn.Module):
    def __init__(
            self,
            vit_lite_model_name="vit_b16",
            vit_lite_proj_dim=512,
            vit_lite_embed_dim=768,

            rnn_type='lstm', # 'lstm', 'gru', 'mamba', 'cross_mamba_film', or 'mamba_spfs'
            rnn_hidden_size=1024,
            rnn_num_layers=1,
            rnn_dropout=0.0,

            fc_hidden_layers=[512],
            teacher_clip_embed_dim=768,
            text_embed_dim=None,
            pred_rank=64,
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
        elif self.rnn_type == 'mamba_spfs':
            self.rnn = MambaSPFS(
                in_dim=vit_lite_embed_dim,
                hidden_dim=rnn_hidden_size,
                clip_dim=teacher_clip_embed_dim,
                pred_rank=pred_rank,
            )
        else:
            raise NotImplementedError(
                f"Unsupported RNN type: {rnn_type}. Choose 'lstm', 'gru', 'mamba', 'cross_mamba_film' or 'mamba_spfs'."
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

        self.last_output = None

    def init_hidden(self, batch_size, device):
        if self.rnn_type in ['mamba', 'cross_mamba_film', 'mamba_spfs']:
            state = self.rnn.init_state(batch_size, device)
            if self.rnn_type == 'mamba_spfs':
                self.rnn.last_hidden = None
                self.last_output = None
            return state
        h0 = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros(self.rnn_num_layers, batch_size, self.rnn_hidden_size).to(device)
            self.last_output = None
            return (h0, c0)
        self.last_output = None
        return h0

    def forward(
        self,
        single_frame_input,
        prev_hidden_state,
        confidence_threshold=0.9,
        max_consecutive_skips=0,
        gamma=None,
        beta=None,
    ):
        """
        Processes a single frame (or a small chunk of frames) and updates the hidden state.

        Note: SPFS is disabled by default (max_consecutive_steps = 0)

        Args:
            single_frame_input (torch.Tensor): Input frame(s) for the ViT-Lite.
                Shape: (B, C, H, W) if student_num_frames_processed_by_vit=1
                Shape: (B, C, T_chunk, H, W) if student_num_frames_processed_by_vit > 1
            prev_hidden_state (tuple or torch.Tensor): Previous hidden state from the RNN.
                For LSTM: (h_prev, c_prev)
                For GRU: h_prev
            confidence_threshold (float): Confidence threshold for SPFS. Default 0.9
            max_consecutive_skips (int): Maximum number of consecutive frames to skip. Default 0
            gamma (torch.Tensor): Used for FiLM
            beta (torch.Tensor): Used for FiLM

        Returns:
            student_embedding (torch.Tensor): The output embedding for the current step.
                                            Shape: (B, teacher_clip_embed_dim)
            current_hidden_state (tuple or torch.Tensor): The updated RNN hidden state.
            spfs_info (dict): Info about SPFS
                skipped (bool): whether the frame was skipped or not
                confidence (float): the confidence level from the predictor
        """
        if len(single_frame_input.shape) == 5:
            single_frame_input = single_frame_input.squeeze(2)

        spfs_info = edict(dict(
            skipped=False,
            confidence=0.0
        ))

        if self.rnn_type == 'mamba_spfs':
            if self.rnn.last_hidden is not None and getattr(self, 'consecutive_skips', 0) < max_consecutive_skips:
                predicted_feature, confidence_logit = self.rnn.predict_next_feat()
                confidence = torch.sigmoid(-confidence_logit).item()
                spfs_info.confidence = confidence

                if confidence > confidence_threshold:
                    spfs_info.skipped = True
                    self.consecutive_skips = getattr(self, 'consecutive_skips', 0) + 1
                    frame_feature = predicted_feature
                else:
                    frame_feature, _ = self.vit_lite.extract_features(single_frame_input)
                    self.consecutive_skips = 0
            else:
                frame_feature, _ = self.vit_lite.extract_features(single_frame_input)
                self.consecutive_skips = 0
        else:
            frame_feature, _ = self.vit_lite.extract_features(single_frame_input)

        if self.rnn_type in ['lstm', 'gru']:
            frame_feature = frame_feature.unsqueeze(1)
            student_embedding, current_hidden_state = self.rnn(frame_feature, prev_hidden_state)
            student_embedding = student_embedding.squeeze(1)
        elif self.rnn_type in ['mamba', 'mamba_spfs']:
            student_embedding, current_hidden_state = self.rnn(frame_feature, prev_hidden_state)
        elif self.rnn_type == 'cross_mamba_film':
            student_embedding, current_hidden_state = self.rnn(frame_feature, prev_hidden_state, gamma, beta)
        else:
            raise NotImplementedError("Assuming Mamba for this review.")

        if self.rnn_type in ['lstm', 'gru']:
            student_embedding = self.output_fc(student_embedding)

        if self.rnn_type == 'mamba_spfs':
            self.last_output = student_embedding
        #print(spfs_info)
        return student_embedding, current_hidden_state, spfs_info
