num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
seed = 42
num_eval_timesteps = 10
mode = "FSDP"

# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop"
)

# webvid
bucket_config = {  # 12s/it
    "512": {1: (1, 72), 49: (1.0, 12), 97: ((1.0, 0.33), 6)},
    # ---
    "480p": {1: (0.5, 48), 49: (0.5, 8), 97: ((0.5, 0.33), 4)},
    # ---
    # "720p": {1: (0.3, 24), 49: (0.3, 4), 97: ((0.3, 0.33), 2)},
}


exp_dir = None

# Model settings
model = dict(
    type="Mochi1",
    from_pretrained=None,
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