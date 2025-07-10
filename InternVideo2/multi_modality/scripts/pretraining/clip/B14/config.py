from configs.data import *
from configs.model import *
import os as __os
from huggingface_hub import hf_hub_download

# ========================= data ==========================
train_corpus = "slim_kinetics"
train_file = "${available_corpus[${train_corpus}]}"  # for lazy evaluation
test_file = dict(act_val=available_corpus["slim_kinetics_act_val"])
test_types = ["act_val"]
num_workers = 2

stop_key = None

# ========================= input ==========================
num_frames = 8
num_frames_test = 8
batch_size = 8      # Use 16 for 5090
batch_size_test = 8 # Use 16 for 5090
max_txt_l = 32

size_t = 224

inputs = dict(
    image_res=size_t,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="all",
        num_frames_test="${num_frames_test}",
        sample_type_test="all",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size_test}", video="${batch_size_test}"),
)

# ========================= model ==========================
HF_REPO = "qingy2024/InternVideo2-B14"

model = dict(
    model_cls="InternVideo2_CLIP_small",
    vision_encoder=dict(
        name="internvideo2",
        in_chans=3,
        patch_size=14,
        img_size=size_t,
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
    streaming_vision_encoder = dict(
        vit_lite_embed_dim = 768,
        rnn_type = 'cross_mamba_film',
        rnn_hidden_size = 1024,
        rnn_num_layers = 3,
        rnn_dropout = 0.0,
        fc_hidden_layers = [768],
        teacher_clip_embed_dim = 768,
        text_embed_dim = 512,
    ),
    mobileclip_type=dict(
        name="mobileclip_b"
    ),
    temp=1 / 100.0,
    temp_min=1 / 100.0,
    use_streaming_vision_align = False,
    freeze_vision=True,
    freeze_mobileclip_vision=True, # Freeze the MobileCLIP vision encoder
    open_vision_clip_projector=False,
    freeze_mobileclip_text=True,
    open_text_projection=False,
    open_text_lora=False,
    vision_ckpt_path=hf_hub_download(repo_id=HF_REPO, filename="internvideo2_vision.pt"),
    load_vision_ckpt_from_internvideo2_stage2=False,
    mobileclip_ckpt_path=hf_hub_download(repo_id=HF_REPO, filename="mobileclip_blt.pt"),
    extra_ckpt_path=hf_hub_download(repo_id=HF_REPO, filename="internvideo2_clip.pt")
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0,
    )
)

optimizer = dict(
    opt="adamW",
    lr=1e-5,
    opt_betas=[0.9, 0.98],
    weight_decay=0.01,
    max_grad_norm=0.7,
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=True, module_names=["streaming_vision_encoder.vit_lite"], lr=2e-6),
)

scheduler = dict(sched="cosine", epochs=1, min_lr_multi=0.01, warmup_epochs=0.1)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

use_half_precision = True
use_bf16 = True
gradient_checkpointing = True

# ========================= wandb ==========================
wandb = dict(
    enable=True,
    entity="qingy2019-conker-mobile-inc-",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="window_iv2",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = './training_outputs_window/'  # output dir
resume = True  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 1
seed = 42

save_latest = False
save_iter = 100
eval_freq_steps = 50

# Evaluation video
eval_video_repo_id = "qingy2024/backflip_train"
eval_video_filename = "1.mp4"

eval_plot_output_dir = 'scripts/pretraining/clip/B14/cosine_sim_graphs'

auto_resume = True
pretrained_path = ""  # path to pretrained model weights, for resume only?

deepspeed = dict(
    enable=True,
    stage=1,
)

# ====================== contrastive distillation =====================
enable_contrastive_distillation = False
contrastive_temperature = 0.07
contrastive_lambda = 0.4
contrastive_warmup_pct = 0.3
contrastive_ramp_iters = 500

# ====================== unfreeze mobileclip =====================
enable_mobileclip_ft = False
unfreeze_mobileclip_pct = 0.5
