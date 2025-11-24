num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
seed = 42
num_eval_timesteps = 10
mode = "FSDP"

# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)

# webvid
bucket_config = {  # 12s/it
   "144p": {1: (1.0, 50), 49: (1.0, 10), 97: (1.0, 5)},
    # # ---
    "256": {1: (0.4, 30), 49: (0.5, 6), 97: (0.5, 3)},
    "240p": {1: (0.3, 30), 49: (0.4, 6), 97: (0.4, 3)},
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
    type="CogVideoXVAE",
    from_pretrained=pretrained_path+"/CogVideoX-2b/vae/",
)


scheduler = dict(
    type="rflow-sd3",
    use_timestep_transform=True,
    sample_method="logit-normal",
)