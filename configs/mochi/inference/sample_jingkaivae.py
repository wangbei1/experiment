resolution = "360p"
aspect_ratio = "9:16"
num_frames = 49
fps = 24
frame_interval = 1
save_fps = 24

save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "OpenSora"
dtype = "bf16"
condition_frame_length = 5
align = None


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

# text_encoder=dict(
#     from_pretrained="/mnt/nas_szy/models/aigc/mt5-xxl/",
#     subfolder="./"
# )
# tokenizer=dict(
#     from_pretrained="/mnt/nas_szy/models/aigc/mt5-xxl/",
#     subfolder="./"
# )

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
    num_sampling_steps=30,
    cfg_scale=7,
)

aes = None
flow = None
