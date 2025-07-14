from configs.data import *
from configs.model import *
from huggingface_hub import hf_hub_download as __hf_hub_download

# ========================= data ==========================
train_corpus = "slim_kinetics"
train_file = "${available_corpus[${train_corpus}]}"
num_workers = 2

# ========================= input ==========================
num_frames = 8
batch_size = 8
size_t = 224

inputs = dict(
    image_res=size_t,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="all",
        random_aug=False,
    ),
    max_txt_l=dict(video=32),
    batch_size=dict(video="${batch_size}"),
)

# ========================= model ==========================
model_repo = "qingy2024/InternVideo2-B14"

model = dict(
    model_cls="InternVideo2_CLIP_small",
    vision_encoder=dict(
        name="internvideo2",
        in_chans=3,
        patch_size=14,
        img_size=size_t,
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
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        layerscale_no_force_fp32=True,
        num_frames="${num_frames}",
        tubelet_size=1,
        align_dim=512,
    ),
    streaming_vision_encoder=dict(
        vit_lite_embed_dim=768,
        rnn_type='stream_mamba',
        rnn_hidden_size=1024,
        rnn_num_layers=3,
        rnn_dropout=0.0,
        fc_hidden_layers=[768],
        teacher_clip_embed_dim=768,
        text_embed_dim=512,
        pred_rank=32,
    ),
    mobileclip_type=dict(name="mobileclip_b"),
    temp=1/100.0,
    temp_min=1/100.0,
    use_streaming_vision_align=False,
    freeze_vision=True,
    freeze_mobileclip_vision=True,
    freeze_mobileclip_text=True,
    vision_ckpt_path=__hf_hub_download(repo_id=model_repo, filename="internvideo2_vision.pt"),
    mobileclip_ckpt_path=__hf_hub_download(repo_id=model_repo, filename="mobileclip_blt.pt"),
    extra_ckpt_path=__hf_hub_download(repo_id=model_repo, filename="internvideo2_clip.pt"),
    train_low_rank_predictor_only=True,
)

optimizer = dict(
    opt="adamW",
    lr=1e-4,
    opt_betas=[0.9, 0.98],
    weight_decay=0.01,
    max_grad_norm=0.7,
)

scheduler = dict(sched="cosine", epochs=1, min_lr_multi=0.01, warmup_epochs=0.1)

evaluate = False
use_half_precision = True
use_bf16 = True
gradient_checkpointing = True

wandb = dict(enable=False, entity="", project="")
dist_url = "env://"
device = "cuda"
mode = "pt"

output_dir = './training_outputs_distillmc/'
resume = False
log_freq = 1
seed = 42

auto_resume = False
pretrained_path = ""

deepspeed = dict(enable=False)
