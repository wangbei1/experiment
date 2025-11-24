num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
seed = 42
num_eval_timesteps = 10
mode = "FSDP"

# Dataset settings
dataset = dict(
    type="VariableVideoTextWithDurationDataset",
    transform_name="resize_crop",
    sample_fps=16,
)

# webvid
bucket_config = {  # 12s/it
    "512": {0: (1, 72), 2: (1.0, 12), 4: ((1.0, 0.33), 6)},
    # ---
    "480p": {0: (0.5, 48), 2: (0.5, 8), 4: ((0.5, 0.33), 4)},
    # ---
    "720p": {0: (0.3, 24), 2: (0.3, 4), 4: ((0.3, 0.33), 2)},
}


exp_dir = None

# Model settings
model = dict(
    type="Flux-3D",
    from_pretrained=None,
    patch_size = 2,
    in_channels = 64,
    num_layers = 19,
    num_single_layers = 38,
    attention_head_dim = 128,
    num_attention_heads = 24,
    joint_attention_dim = 4096,
    pooled_projection_dim = 768,
    guidance_embeds = False,
    axes_dims_rope = [16, 56, 56]
)
pretrained_path = "/mnt/workspace/workgroup/moyuan.yty/.cache/flux"
text_encoder=dict(
    from_pretrained=pretrained_path+"/FLUX.1-dev-diffusers/",
    subfolder="text_encoder"
)
text_encoder_2=dict(
    from_pretrained=pretrained_path+"/FLUX.1-dev-diffusers/",
    subfolder="text_encoder_2"
)
tokenizer=dict(
    from_pretrained=pretrained_path+"/FLUX.1-dev-diffusers/",
    subfolder="tokenizer"
)
tokenizer_2=dict(
    from_pretrained=pretrained_path+"/FLUX.1-dev-diffusers/",
    subfolder="tokenizer_2"
)

vae = dict(
    type="JINGKAI_VAE",
    pretrained_spatial_vae_path=pretrained_path+"/FLUX.1-dev-diffusers/",
    from_pretrained=pretrained_path+"/jingkai_vae/checkpoint-stage3-final.pth",
)

scheduler = dict(
    type="rflow-sd3",
    use_timestep_transform=True,
    sample_method="logit-normal",
)