resolution = "1024"
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
    type="Flux-base",
    from_pretrained="/home/dufei.df/workgroup_shanghai/dufei.df/models/aigc/FLUX.1-schnell-diffusers/transformer_onefile_bfp16/diffusion_pytorch_model.bfp16.safetensors",
    patch_size = 1,
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

aes = None
flow = None
