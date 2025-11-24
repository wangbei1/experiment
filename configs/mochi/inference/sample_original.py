resolution = "360p"
aspect_ratio = "9:16"
num_frames = 49
fps = 30
frame_interval = 1
save_fps = 30

save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "OpenSora"
dtype = "bf16"
condition_frame_length = 5
align = None


model = dict(
    type="Mochi1",
    from_pretrained="/home/dufei.df/models/aigc/mochi-1-preview/dit.safetensors",
    depth=48,
    patch_size=2,
    num_heads=24,
    hidden_size_x=3072,
    hidden_size_y=1536,
    mlp_ratio_x=4.0,
    mlp_ratio_y=4.0,
    in_channels=12,
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
    from_pretrained="/home/dufei.df/models/aigc/t5-v1_1-xxl/",
    subfolder="./"
)
tokenizer=dict(
    from_pretrained="/home/dufei.df/models/aigc/t5-v1_1-xxl/",
    subfolder="./"
)

vae = dict(
    from_pretrained="/home/dufei.df/models/aigc/mochi-1-preview/vae.safetensors",
)

scheduler = dict(
    type="rflow-mochi",
    num_sampling_steps=50,
    cfg_scale=4.5,
)

aes = None
flow = None
