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

pretrained_path = "/mnt/workspace/workgroup/moyuan.yty/.cache/flux"
model = dict(
    type="Flux-3D",
    from_pretrained=pretrained_path+"/015-Flux-3D/epoch0-global_step2000/ema",
    patch_size = 2,
    in_channels = 64,
    num_layers = 19,
    num_single_layers = 38,
    attention_head_dim = 128,
    num_attention_heads = 24,
    joint_attention_dim = 4096,
    pooled_projection_dim = 768,
    guidance_embeds = False,
    axes_dims_rope = [16, 56, 56],
)

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
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = None
flow = None
