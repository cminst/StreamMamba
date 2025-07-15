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
max_txt_l = 32
size_t = 224

inputs = dict(
    image_res=size_t,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="all",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
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
        depth=12,
        use_flash_attn=True,
        use_fused_rmsnorm=True,
        use_fused_mlp=True,
        fused_mlp_heuristic=1,
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        layerscale_no_force_fp32=True,
        num_frames=num_frames,
        tubelet_size=1,
        sep_pos_embed=False,
        use_checkpoint=False,
        checkpoint_num=0,
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
    ),
    mobileclip_type=dict(name='mobileclip_b'),
    temp=1/100.0,
    temp_min=1/100.0,
    use_streaming_vision_align=False,
    freeze_vision=True,
    freeze_mobileclip_vision=True,
    freeze_mobileclip_text=True,
    freeze_mamba=False,
    freeze_prediction_head=False,
    freeze_confidence_head=False,
    vision_ckpt_path=__hf_hub_download(repo_id=model_repo, filename='internvideo2_vision.pt'),
    mobileclip_ckpt_path=__hf_hub_download(repo_id=model_repo, filename='mobileclip_blt.pt'),
    extra_ckpt_path=__hf_hub_download(repo_id=model_repo, filename='internvideo2_clip.pt'),
)

optimizer = dict(
    opt='adamW',
    lr=1e-5,
    opt_betas=[0.9, 0.98],
    weight_decay=0.01,
)

scheduler = dict(sched='cosine', epochs=2, min_lr_multi=0.01, warmup_epochs=0.1)

evaluate = False
device = 'cuda'
mode = 'pt'
output_dir = './training_outputs_spfs/'
wandb = dict(enable=False)
seed = 42
