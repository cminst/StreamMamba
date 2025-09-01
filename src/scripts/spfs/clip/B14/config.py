from configs.data import *
from configs.model import *
from huggingface_hub import hf_hub_download as __hf_hub_download

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
batch_size = 8
batch_size_test = 8
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
model_repo = "cminst/StreamMamba"

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
    streaming_vision_encoder=dict(
        vit_lite_embed_dim=768,
        rnn_type='mamba_spfs',
        rnn_hidden_size=1024,
        rnn_num_layers=3,
        rnn_dropout=0.0,
        fc_hidden_layers=[768],
        teacher_clip_embed_dim=768,
        text_embed_dim=512,
        pred_rank=64, # or None for a full linear layer
    ),
    mobileclip_type=dict(name='mobileclip_b'),
    temp=1 / 100.0,
    temp_min=1 / 100.0,
    open_vision_clip_projector=False,
    open_text_projection=False,
    open_text_lora=False,

    use_streaming_vision_align = False,
    freeze_vision=True,
    freeze_mobileclip_vision=True,
    freeze_mobileclip_text=True,
    freeze_mamba=False,
    freeze_prediction_head=False,
    freeze_confidence_head=False,
    load_vision_ckpt_from_internvideo2_stage2=False,
    vision_ckpt_path=__hf_hub_download(repo_id=model_repo, filename='internvideo2_vision.pt'),
    mobileclip_ckpt_path=__hf_hub_download(repo_id=model_repo, filename='mobileclip_blt.pt'),
    extra_ckpt_path=__hf_hub_download(repo_id=model_repo, filename='internvideo2_clip.pt'),
)

criterion = dict(loss_weight=dict(vtc=1.0))

optimizer = dict(
    opt="adamW",
    lr=1e-5,
    opt_betas=[0.9, 0.98],
    weight_decay=0.01,
    max_grad_norm=0.7,
    different_lr=dict(enable=False, module_names=["streaming_vision_encoder.vit_lite"], lr=2e-6),
)

scheduler = dict(sched='cosine', epochs=2, min_lr_multi=0.01, warmup_epochs=0.1)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,
)

use_half_precision = True
use_bf16 = True
gradient_checkpointing = True

# =================== spfs hyperparams =====================

lambda_calib = 1.0
lambda_skip = 0.0
calibration_loss_fn = 'mse' # or 'bce'

# ========================= wandb ==========================
wandb = dict(
    enable=True,
    entity="qingy2019-conker-mobile-inc-",
    project="window_iv2",
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = './train_outputs_spfs/'
resume = True
debug = False
log_freq = 1
seed = 42

save_latest = False
save_iter = 500

auto_resume = False
pretrained_path = ""

deepspeed = dict(
    enable=True,
    stage=1,
)
