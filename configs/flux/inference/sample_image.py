resolution = "512"
aspect_ratio = "1:1"
num_frames = 1
fps = 1
frame_interval = 1
save_fps = 1

save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "OpenSora"
dtype = "bf16"
condition_frame_length = 5
align = None

model = dict(
    type="Flux-3D",
    from_pretrained="/home/dufei.df/models/aigc/FLUX-mt5-xxl/transformer/model.safetensors",
    patch_size = 2,
    in_channels = 64,
    num_layers = 19,
    num_single_layers = 38,
    attention_head_dim = 128,
    num_attention_heads = 24,
    joint_attention_dim = 4096,
    pooled_projection_dim = 768,
    guidance_embeds = False,
    t5_embedder=True,
    axes_dims_rope = [16, 56, 56],
)

text_encoder=dict(
    from_pretrained="/home/dufei.df/models/aigc/FLUX-mt5-xxl/",
    subfolder="text_encoder"
)
text_encoder_2=dict(
    from_pretrained="/home/dufei.df/models/aigc/FLUX-mt5-xxl/",
    subfolder="text_encoder_2"
)
tokenizer=dict(
    from_pretrained="/home/dufei.df/models/aigc/FLUX-mt5-xxl/",
    subfolder="tokenizer"
)
tokenizer_2=dict(
    from_pretrained="/home/dufei.df/models/aigc/FLUX-mt5-xxl/",
    subfolder="tokenizer_2"
)

vae = dict(
    type="JINGKAI_VAE",
    pretrained_spatial_vae_path="/home/dufei.df/models/aigc/FLUX.1-dev-diffusers/",
    from_pretrained="/home/dufei.df/models/aigc/video_vae/checkpoint-stage4-final.pth",
    scaling_factor=0.9480883479118347,
    shift_factor=0.04052285850048065,
)

# vae=dict(
#     from_pretrained="/home/dufei.df/models/aigc/FLUX.1-dev-diffusers/",
#     subfolder="vae"
# )

scheduler = dict(
    type="rflow-sd3",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = None
flow = None
