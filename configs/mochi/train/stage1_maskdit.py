# Dataset settings
# dataset = dict(
#     type="VariableVideoTextPerRankDataset",
#     transform_name="resize_crop",
#     data_path="/mnt/nas_szy/huaniu_workspace/Data/video_data_360M_filtered_240p_4s_bs4_node32/split_data_{}.feather",
#     sample_fps=16,
# )

# dataset = dict(
#     type="VariableVideoTextPerRankDataset",
#     transform_name="resize_crop",
#     data_path="/mnt/nas_szy/huaniu_workspace/Data/video_data_360M_filtered_240p_4s_bs4_node32/split_data_{}.feather",
#     image_data_path="/mnt/nas_szy/huaniu_workspace/Data/image_data_vcg_joy_240p_bs4_node32/split_data_{}.feather",
#     image_percent=0.1,
#     sample_fps=16,
# )

# dataset = dict(
#     type="VariableVideoTextPerRankDataset",
#     transform_name="resize_crop",
#     data_path="/mnt/nas_szy/huaniu_workspace/Data/image_data_vcg_joy_240p_bs4_node32/split_data_{}.feather",
#     sample_fps=16,
# )


eval_dataset = dict(
    type="VariableVideoTextDataset",
    data_path="/mnt/workspace/workgroup/moyuan.yty/Meta/hd_720p_val.csv",
    transform_name="resize_crop",
)

eval_bucket_config = {  # 12s/it
    "512": {1: (1, 72), 49: (1.0, 12), 97: ((1.0, 0.33), 6)},
    # ---
    "480p": {1: (0.5, 48), 49: (0.5, 8), 97: ((0.5, 0.33), 4)},
    # ---
    # "720p": {0: (0.3, 24), 49: (0.3, 4), 97: ((0.3, 0.33), 2)},
}

dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop"
)

# webvid
bucket_config = {  # 12s/it
    "512": {1: (1, 72), 49: (1.0, 6), 97: ((1.0, 0.33), 3)},
    # ---
    "480p": {1: (0.5, 48), 49: (0.5, 4), 97: ((0.5, 0.33), 2)},
    # ---
    # "720p": {1: (0.3, 24), 49: (0.3, 2), 97: ((0.3, 0.33), 1)},
}

grad_checkpoint = True

# Acceleration settings
num_workers = 8
dtype = "bf16"
mode = "FSDP"
sp_degree = 1
monitor = False

exp_flag = "jingkaivae-maskdit"
maskdit = True
fulltoken = False
unpatchify_loss = False

# Model settings
model = dict(
    type="Mochi1",
    from_pretrained="/mnt/nas_szy/aigc/VidGen_acc/VidGen/pretrained_weights/mochi_jingkaivae_step9500_ema.safetensors",#"/mnt/nas_szy/models/aigc/mochi-1-preview/dit_16channel_vae.safetensors", 
    depth=48,
    patch_size=2,
    num_heads=24,
    hidden_size_x=3072,
    hidden_size_y=1536,
    mlp_ratio_x=4.0,
    mlp_ratio_y=4.0,
    in_channels=16,
    qk_norm=True,
    qkv_bias=False,
    out_bias=True,
    patch_embed_bias=True,
    timestep_mlp_bias=True,
    timestep_scale=1000.0,
    t5_feat_dim=4096,
    t5_token_length=256,
    rope_theta=10000.0,
    maskdit=maskdit,
    decoder_depth=4,
)

text_encoder=dict(
    from_pretrained="/mnt/nas_szy/models/aigc/t5-v1_1-xxl/",
    subfolder="./"
)
tokenizer=dict(
    from_pretrained="/mnt/nas_szy/models/aigc/t5-v1_1-xxl/",
    subfolder="./"
)

vae = dict(
    type="JINGKAI_VAE",
    pretrained_spatial_vae_path="/mnt/nas_szy/models/aigc/FLUX.1-dev-diffusers/",
    from_pretrained="/mnt/nas_szy/models/aigc/video_vae/checkpoint-stage4-final.pth",
    scaling_factor=0.9480883479118347,
    shift_factor=0.04052285850048065,
)

scheduler = dict(
    type="rflow-mochi",
    use_timestep_transform=True,
    sample_method="logit-normal",
)

# Temporal mask settings
# temporal_mask_ratios = {
#     "random": 0.05,
#     "intepolate": 0.005,
#     "quarter_random": 0.005,
#     "quarter_head": 0.005,
#     "quarter_tail": 0.005,
#     "quarter_head_tail": 0.005,
#     "image_random": 0.025,
#     "image_head": 0.05,
#     "image_tail": 0.025,
#     "image_head_tail": 0.025,
# }
# Spatial mask settings
if maskdit and not fulltoken:
    spatial_mask_ratio = 0.5
    mae_loss_coef = 0.1

prompt_uncond_prob=0.1 #for cfg
mask_ratios=None
# Log settings
seed = 42
outputs = "outputs"
wandb = False
epochs = 1000
log_every = 1
ckpt_every = 500
eval_every = 500

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-5
ema_decay = 0.99
adam_eps = 1e-15
weight_decay= 1e-4
warmup_steps = 200
