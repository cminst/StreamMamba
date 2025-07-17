from configs.data import *
from configs.model import *
from huggingface_hub import hf_hub_download as __hf_hub_download

# ========================= Main Settings ==========================
# The output directory where embed_b*****.pt files will be saved.
output_dir = './train_outputs_spfs_embeddings/'

num_gpus_for_computation = 4

# ========================= data ==========================
train_corpus = "slim_kinetics"
train_file = "${available_corpus[${train_corpus}]}"
num_workers = 4

# ========================= input ==========================
num_frames = 8
batch_size = 8

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="all",
        num_frames_test="${num_frames}",
        sample_type_test="all",
        random_aug=False,
    ),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size}", video="${batch_size}"),
)

# ========================= model ==========================
model_repo = "qingy2024/InternVideo2-B14"
model = dict(
    model_cls="InternVideo2_CLIP_small",
    vision_encoder=dict(
        name="internvideo2",
        in_chans=3,
        patch_size=14,
        img_size=224,
        qkv_bias=False,
        drop_path_rate=0.,
        head_drop_path_rate=0.,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4,
        init_values=0.1,
        qk_normalization=True,
        depth=12,
        use_flash_attn=True,
        use_fused_rmsnorm=True,
        use_fused_mlp=True,
        fused_mlp_heuristic=1,
        drop_cls_token=False,
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        layerscale_no_force_fp32=True,
        num_frames=8,
        tubelet_size=1,
        sep_pos_embed=False,
        use_checkpoint=False,
        checkpoint_num=0,
        align_dim=512,
    ),
    streaming_vision_encoder=dict(
        vit_lite_embed_dim=768,
        rnn_type='mamba_spfs',
        rnn_hidden_size=1024,
        rnn_num_layers=3,
        rnn_dropout=0.0,
        fc_hidden_layers=[768],
        teacher_clip_embed_dim=768,
        text_embed_dim=512,
        pred_rank=32,
    ),
    mobileclip_type=dict(name='mobileclip_b'),
    vision_ckpt_path=__hf_hub_download(repo_id=model_repo, filename='internvideo2_vision.pt'),
    mobileclip_ckpt_path=__hf_hub_download(repo_id=model_repo, filename='mobileclip_blt.pt'),
    extra_ckpt_path=__hf_hub_download(repo_id=model_repo, filename='internvideo2_clip.pt'),
)

# ========================= environment ==========================
use_half_precision = True
use_bf16 = True
dist_url = "env://"
device = "cuda"
mode = "pt"
seed = 42
log_freq = 10

# We disable deepspeed as we are manually managing devices for inference.
# DDP is still used for the distributed sampler in the dataloader.
deepspeed = dict(enable=False)

# Unused training params
evaluate = True
optimizer = {}
scheduler = {}
wandb = dict(enable=False)
