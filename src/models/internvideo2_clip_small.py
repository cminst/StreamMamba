import logging
import os
import json

import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from .backbones.internvideo2 import InternVideo2, TextTransformer, ClipTokenizer, StreamMamba
from .criterions import VTC_VTM_Loss
from .utils import unwrap_state_dict

logger = logging.getLogger(__name__)


class InternVideo2_CLIP_small(nn.Module):
    def __init__(self, config, tokenizer=None, is_pretrain=True):
        """
        Initialize the InternVideo2_CLIP_small model.

        Args:
            config (dict): Configuration dictionary containing model parameters.
            tokenizer (ClipTokenizer, optional): Tokenizer for text processing. Defaults to None.
            is_pretrain (bool, optional): Flag indicating if the model is in pretraining mode. Defaults to True.
        """
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.is_pretrain = is_pretrain

        # Load MobileCLIP encoder configuration
        self.mobileclip_cfg = json.load(
            open(os.path.join(
                "./models/backbones/internvideo2/mobileclip/configs/" +
                f"{self.config.model.mobileclip_type.name}.json"))
        )
        if tokenizer is None:
            self.tokenizer = ClipTokenizer(self.mobileclip_cfg)

        # Build vision encoder
        self.vision_encoder = self.build_vision_encoder()

        # Define vision alignment layers
        self.vision_align = nn.Sequential(
            nn.LayerNorm(self.config.model.vision_encoder.clip_embed_dim),
            nn.Linear(
                self.config.model.vision_encoder.clip_embed_dim,
                self.config.model.vision_encoder.align_dim
            )
        )

        if config.model.use_streaming_vision_align:
            self.streaming_vision_align = nn.Sequential(
                nn.LayerNorm(self.config.model.vision_encoder.clip_embed_dim),
                nn.Linear(
                    self.config.model.vision_encoder.clip_embed_dim,
                    self.config.model.vision_encoder.align_dim
                )
            )

        # Build StreamMamba for distillation
        self.streaming_vision_encoder = self.build_streaming_vision_encoder()

        # Build text encoder
        self.text_encoder = self.build_text_encoder(
            cfg=self.mobileclip_cfg['text_cfg'],
            projection_dim=self.mobileclip_cfg["embed_dim"]
        )

        # Initialize temperature parameter
        self.temp = nn.parameter.Parameter(torch.ones([]) * config.model.temp)
        self.temp_min = config.model.temp_min

        # Freeze model parameters if specified in the config
        if self.config.model.freeze_vision:
            self.freeze_vision()

        if self.config.model.freeze_mobileclip_vision:
            self.freeze_mobileclip_vision()

        if self.config.model.freeze_mobileclip_text:
            self.freeze_mobileclip_text()

        if getattr(self.config.model, "freeze_mamba", False):
            self.freeze_mamba()

        if getattr(self.config.model, "freeze_prediction_head", False):
            self.freeze_prediction_head()

        if getattr(self.config.model, "freeze_confidence_head", False):
            self.freeze_confidence_head()

        # Define image transformation pipeline
        img_size = self.config.model.vision_encoder.img_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.Lambda(lambda x: x.float().div(255.0)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # Load pretrained models
        self.load_checkpoint(
            vision_ckpt_path=config.model.vision_ckpt_path,
            mobileclip_ckpt_path=config.model.mobileclip_ckpt_path,
            extra_ckpt_path=config.model.get("extra_ckpt_path", None)
        )

        # Initialize loss criterion
        self.clip_loss = VTC_VTM_Loss(False)

    def no_weight_decay(self):
        ret = {"temp"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        # no weight decay for LLM if training
        ret.update(
            {"text_encoder." + k for k, _ in self.text_encoder.named_parameters()}
        )

        return ret

    @torch.no_grad()
    def clip_contrastive_temperature(self):
        """Seems only used during pre-training"""
        self.temp.clamp_(min=self.temp_min)

    def forward(self, image, text, idx):
        """forward and calculate loss.

        Args:
            image (torch.Tensor): The input images. Shape: [B,T,C,H,W].
            text (dict): TODO
            idx (torch.Tensor): TODO

        Returns: TODO

        """
        self.clip_contrastive_temperature()
        vision_embeds = self.encode_vision(image)
        text_embeds = self.encode_text(text)

        # Video-text contrastive (VTC) loss
        loss_vtc = self.clip_loss.vtc_loss(
            vision_embeds, text_embeds, idx, self.temp, all_gather=True
        )

        return dict(
            loss_vtc=loss_vtc,
        )

    def encode_vision(self, image):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The features of all patches. Shape: [B,C].
        """

        T = image.shape[1]
        use_image = True if T == 1 else False

        image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]
        vision_embeds = self.vision_encoder(image, use_image=use_image)

        vision_embeds_aligned = self.vision_align(vision_embeds)

        return vision_embeds_aligned

    def get_vid_feat(self, frames: torch.Tensor):
        with torch.no_grad():
            vfeat = self.encode_vision(frames)

            # vfeat = self.vision_proj(vfeat)
            vfeat /= vfeat.norm(dim=-1, keepdim=True)

        return vfeat

    def encode_streaming_vision(
        self,
        image,
        prev_hidden_state,
        confidence_threshold=0.9,
        max_consecutive_skips=0,
        gamma=None,
        beta=None,
    ):
        """Encode image/video frames using the streaming ViT.

        Note: SPFS is disabled by default (max_consecutive_steps = 0)

        Args:
            image (torch.Tensor): The input images.
            prev_hidden_state (tuple or torch.Tensor): Previous hidden state from the RNN.
                For LSTM: (h_prev, c_prev)
                For GRU: h_prev
            confidence_threshold (float): Confidence threshold for SPFS. Default 0.9
            max_consecutive_skips (int): Maximum number of consecutive frames to skip. Default 0
            gamma (torch.Tensor, optional): FiLM scale parameters for conditioning the
                streaming encoder. ``None`` if FiLM is not used.
            beta (torch.Tensor, optional): FiLM shift parameters for conditioning the
                streaming encoder. ``None`` if FiLM is not used.

        Returns:
            tuple: ``(vision_embeds, new_hidden_state)`` where ``vision_embeds`` is
            the encoded features for the current frame(s) and ``new_hidden_state`` is
            the updated hidden state.
        """

        assert len(image.shape) in [4, 5], f"Invalid dimension: {image.shape}"

        vision_embeds, new_hidden_state, spfs_info = self.streaming_vision_encoder(
            image,
            prev_hidden_state=prev_hidden_state,
            confidence_threshold=confidence_threshold,
            max_consecutive_skips=max_consecutive_skips,
            gamma=gamma,
            beta=beta,
        )

        if self.config.model.use_streaming_vision_align:
            vision_embeds_aligned = self.streaming_vision_align(vision_embeds)
        else:
            vision_embeds_aligned = self.vision_align(vision_embeds)

        return vision_embeds_aligned, new_hidden_state, spfs_info

    def get_streaming_vid_feat(
        self,
        frames: torch.Tensor,
        prev_hidden_state,
        confidence_threshold=0.9,
        max_consecutive_skips=0,
        gamma=None,
        beta=None,
    ):
        """Return features for a single frame using the streaming ViT.

        Note: SPFS is disabled by default (max_consecutive_steps = 0)

        Args:
            frames (torch.Tensor): Input frame(s) for the ViT-Lite.
                Shape ``(B, C, H, W)`` for single-frame input or
                ``(B, C, T_chunk, H, W)`` when processing multiple frames at once.
            prev_hidden_state (tuple or torch.Tensor): Previous hidden state from the RNN.
                For LSTM: ``(h_prev, c_prev)``; for GRU: ``h_prev``.
            confidence_threshold (float): Confidence threshold for SPFS. Default 0.9
            max_consecutive_skips (int): Maximum number of consecutive frames to skip. Default 0
            gamma (torch.Tensor, optional): FiLM scale parameters for conditioning the
                streaming encoder. ``None`` if FiLM is not used.
            beta (torch.Tensor, optional): FiLM shift parameters for conditioning the
                streaming encoder. ``None`` if FiLM is not used.

        Returns:
            tuple: ``(vfeat, new_hidden_state)`` where ``vfeat`` is the normalised
            video feature embedding and ``new_hidden_state`` is the updated RNN state.
        """
        with torch.no_grad():
            vfeat, new_hidden_state, spfs_info = self.encode_streaming_vision(
                frames,
                prev_hidden_state=prev_hidden_state,
                confidence_threshold=confidence_threshold,
                max_consecutive_skips=max_consecutive_skips,
                gamma=gamma,
                beta=beta,
            )

            # vfeat = self.vision_proj(vfeat)
            vfeat /= vfeat.norm(dim=-1, keepdim=True)

        return vfeat, new_hidden_state, spfs_info

    def encode_text(self, text):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,C].

        """
        text_embeds = self.text_encoder(text)
        return text_embeds

    def get_txt_feat(self,
                     text: str):
        """get the text features for the given text."""
        t_original = text
        with torch.no_grad():
            text = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_txt_l,
                return_tensors="pt",).to(self.config.device)
            tfeat = self.encode_text(text)
            # tfeat = self.text_proj(tfeat)
            tfeat /= tfeat.norm(dim=-1, keepdim=True)
        return tfeat

    def build_vision_encoder(self):
        """
        Build the InternVideo2 model.

        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.
        """

        config = self.config.model.vision_encoder
        vision_encoder = InternVideo2(
            in_chans=config.in_chans,
            patch_size=config.patch_size,
            img_size=config.img_size,
            qkv_bias=config.qkv_bias,
            drop_path_rate=config.drop_path_rate,
            head_drop_path_rate=config.head_drop_path_rate,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            init_values=config.init_values,
            qk_normalization=config.qk_normalization,
            depth=config.depth,
            use_flash_attn=False,
            use_fused_rmsnorm=False,
            use_fused_mlp=False,
            fused_mlp_heuristic=config.fused_mlp_heuristic,
            attn_pool_num_heads=config.attn_pool_num_heads,
            clip_embed_dim=config.clip_embed_dim,
            layerscale_no_force_fp32=config.layerscale_no_force_fp32,
            num_frames=config.num_frames,
            tubelet_size=config.tubelet_size,
            sep_pos_embed=config.sep_pos_embed,
            use_checkpoint=config.use_checkpoint,
            checkpoint_num=config.checkpoint_num,
        )
        return vision_encoder

    def build_text_encoder(self, cfg, projection_dim):
        """Build the text encoder from MobileCLIP.
        Returns: nn.Module. The text encoder

        """
        text_encoder = TextTransformer(cfg, projection_dim)

        return text_encoder

    def build_streaming_vision_encoder(self):
        """
        Build the StreamMamba model.

        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.
        """

        config = self.config.model.streaming_vision_encoder

        streaming_vision_encoder = StreamMamba(
            vit_lite_model_name=self.mobileclip_cfg["image_cfg"]["model_name"],
            vit_lite_proj_dim=self.mobileclip_cfg["embed_dim"], # Projection dimension
            vit_lite_embed_dim=config.vit_lite_embed_dim, # Output dimension
            rnn_type = config.rnn_type,
            rnn_hidden_size = config.rnn_hidden_size,
            rnn_num_layers = config.rnn_num_layers,
            rnn_dropout = config.rnn_dropout,
            fc_hidden_layers = config.fc_hidden_layers,
            teacher_clip_embed_dim = config.teacher_clip_embed_dim,
            text_embed_dim = self.mobileclip_cfg["embed_dim"],
            pred_rank = config.get('pred_rank', None), # Rank for SPFS
        )

        return streaming_vision_encoder

    def load_checkpoint(self, vision_ckpt_path=None, mobileclip_ckpt_path=None, extra_ckpt_path=None):
        assert vision_ckpt_path is not None, "No vision_encoder checkpoint"
        assert mobileclip_ckpt_path is not None, "No mobileclip checkpoint (for text_encoder and single_vision_encoder)"

        new_ckpt = {}

        # load vision_encoder (InternVideo2 part)
        logger.info(f"Load vision_encoder checkpoint from {vision_ckpt_path}")
        vision_ckpt = unwrap_state_dict(torch.load(vision_ckpt_path, map_location='cpu'))

        if self.config.model.get('load_vision_ckpt_from_internvideo2_stage2', False):
            from .backbones.internvideo2.pos_embed import interpolate_pos_embed
            orig_t_size = self.config.model.get('vision_ckpt_t_size', 4)
            interpolate_pos_embed(vision_ckpt, self.vision_encoder, orig_t_size=orig_t_size) # 4 for InternVideo2 stage2
            for k, v in vision_ckpt.items():
                if k.startswith('vision_encoder.'):
                    if 'clip_decoder' in k or 'final_clip_decoder' in k:
                        continue
                    elif 'clip_pos_embed' in k or 'clip_img_pos_embed' in k or 'img_pos_embed' in k :
                        continue
                    else:
                        new_ckpt[k] = v
                else:
                    continue
        else:
            for k, v in vision_ckpt.items():
                # These keys are from the InternVideo2 checkpoint structure
                if k.startswith('clip_decoder.') or k.startswith('mae_decoder.') or k.startswith('final_clip_decoder.'):
                    continue
                elif k in ['clip_pos_embed', 'mae_pos_embed']:
                    continue
                else:
                    # Prefix with 'vision_encoder.' for the InternVideo2 part
                    new_k = 'vision_encoder.' + k
                    new_ckpt[new_k] = v

        # load text_encoder and single_vision_encoder (MobileCLIP parts)
        logger.info(f"Load mobileclip checkpoint from {mobileclip_ckpt_path}")
        mobileclip_ckpt = unwrap_state_dict(torch.load(mobileclip_ckpt_path, map_location='cpu'))

        for k, v in mobileclip_ckpt.items():
            if k.startswith('text_encoder.'):
                new_ckpt[k] = v
            elif k.startswith('image_encoder.'):
                new_k = 'streaming_vision_encoder.vit_lite.' + k[len('image_encoder.model.'):]
                new_ckpt[new_k] = v

        # load extra checkpoint
        if extra_ckpt_path is not None:
            logger.info(f"Load extra checkpoint from {extra_ckpt_path}")
            extra_ckpt = unwrap_state_dict(torch.load(extra_ckpt_path, map_location='cpu'))

            for k, v in extra_ckpt.items():
                new_ckpt[k] = v

        msg = self.load_state_dict(new_ckpt, strict=False)
        logger.info(msg)

    def predict_label(self,
                      vid_feat: torch.Tensor,
                      txt_feat: torch.Tensor,
                      top: int=5):
        label_probs = (100.0 * vid_feat @ txt_feat.T)
        top_probs, top_labels = label_probs.float().cpu().topk(top, dim=-1)
        return top_probs, top_labels

    # ========== Parameter Freezing Utilities ==========

    def _set_requires_grad(self, module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def freeze_vision(self):
        for name, p in self.vision_encoder.named_parameters():
            if self.config.model.open_vision_clip_projector and name.startswith('clip_projector'):
                continue
            p.requires_grad = False
        for name, p in self.vision_align.named_parameters():
            if self.config.model.open_vision_clip_projector and name.startswith('clip_projector'):
                continue
            p.requires_grad = False

    def unfreeze_vision(self):
        self._set_requires_grad(self.vision_encoder, True)
        self._set_requires_grad(self.vision_align, True)

    def freeze_mobileclip_vision(self):
        self._set_requires_grad(self.streaming_vision_encoder.vit_lite, False)

    def unfreeze_mobileclip_vision(self):
        self._set_requires_grad(self.streaming_vision_encoder.vit_lite, True)

    def freeze_mobileclip_text(self):
        for name, p in self.text_encoder.named_parameters():
            if self.config.model.open_text_projection and name.startswith('projection_layer'):
                continue
            p.requires_grad = False

    def unfreeze_mobileclip_text(self):
        self._set_requires_grad(self.text_encoder, True)

    def freeze_mamba(self):
        for name, p in self.streaming_vision_encoder.rnn.named_parameters():
            # Skip predictor and confidence heads
            if name.startswith('pred_') or name.startswith('logvar'):
                continue
            p.requires_grad = False

    def unfreeze_mamba(self):
        for name, p in self.streaming_vision_encoder.rnn.named_parameters():
            # Skip predictor and confidence heads
            if name.startswith('pred_') or name.startswith('logvar'):
                continue
            p.requires_grad = True

    def freeze_prediction_head(self):
        if hasattr(self.streaming_vision_encoder.rnn, 'pred_U'):
            self.streaming_vision_encoder.rnn.pred_U.requires_grad = False
        if hasattr(self.streaming_vision_encoder.rnn, 'pred_V'):
            self._set_requires_grad(self.streaming_vision_encoder.rnn.pred_V, False)
        if hasattr(self.streaming_vision_encoder.rnn, 'pred_head'):
            self._set_requires_grad(self.streaming_vision_encoder.rnn.pred_head, False)

    def unfreeze_prediction_head(self):
        if hasattr(self.streaming_vision_encoder.rnn, 'pred_U'):
            self.streaming_vision_encoder.rnn.pred_U.requires_grad = True
        if hasattr(self.streaming_vision_encoder.rnn, 'pred_V'):
            self._set_requires_grad(self.streaming_vision_encoder.rnn.pred_V, True)
        if hasattr(self.streaming_vision_encoder.rnn, 'pred_head'):
            self._set_requires_grad(self.streaming_vision_encoder.rnn.pred_head, True)

    def freeze_confidence_head(self):
        if hasattr(self.streaming_vision_encoder.rnn, 'logvar'):
            self._set_requires_grad(self.streaming_vision_encoder.rnn.logvar, False)

    def unfreeze_confidence_head(self):
        if hasattr(self.streaming_vision_encoder.rnn, 'logvar'):
            self._set_requires_grad(self.streaming_vision_encoder.rnn.logvar, True)
