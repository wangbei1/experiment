# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
)

# backup
# bucket_config = {  # 20s/it
#     "144p": {1: (1.0, 100), 51: (1.0, 30), 102: (1.0, 20), 204: (1.0, 8), 408: (1.0, 4)},
#     # ---
#     "256": {1: (0.5, 100), 51: (0.3, 24), 102: (0.3, 12), 204: (0.3, 4), 408: (0.3, 2)},
#     "240p": {1: (0.5, 100), 51: (0.3, 24), 102: (0.3, 12), 204: (0.3, 4), 408: (0.3, 2)},
#     # ---
#     "360p": {1: (0.5, 60), 51: (0.3, 12), 102: (0.3, 6), 204: (0.3, 2), 408: (0.3, 1)},
#     "512": {1: (0.5, 60), 51: (0.3, 12), 102: (0.3, 6), 204: (0.3, 2), 408: (0.3, 1)},
#     # ---
#     "480p": {1: (0.5, 40), 51: (0.3, 6), 102: (0.3, 3), 204: (0.3, 1), 408: (0.0, None)},
#     # ---
#     "720p": {1: (0.2, 20), 51: (0.3, 2), 102: (0.3, 1), 204: (0.0, None)},
#     "1024": {1: (0.1, 20), 51: (0.3, 2), 102: (0.3, 1), 204: (0.0, None)},
#     # ---
#     "1080p": {1: (0.1, 10)},
#     # ---
#     "2048": {1: (0.1, 5)},
# }

# webvid
bucket_config = {  # 12s/it
   "144p": {1: (1.0, 80), 49: (1.0, 20), 97: ((1.0, 0.5), 10)},
    # # ---
    "256": {1: (0.4, 50), 49: (0.5, 6), 97: ((0.5, 0.33), 3)},
    "240p": {1: (0.3, 50), 49: (0.4, 6), 97: ((0.4, 0.33), 3)},
    # # "144p": {49: (1.0, 100)}
    # # ---
    # "360p": {1: (0.2, 141), 49: (0.15, 8), 97: (0.15, 4)},
    # # "144p": {49: (1.0, 100)}
    # # ---
    # "360p": {1: (0.2, 150), 51: (0.15, 5), 102: (0.15, 3)},
    # "144p": {1: (1.0, 10)},
    # # ---
    # "256": {1: (0.4, 5)},
    # "240p": {1: (0.3, 5)},
    # # ---
    # "360p": {1: (0.2, 3)},
    # "512": {1: (0.1, 141)},
    # # ---
    # "480p": {1: (0.1, 89)},
    # # ---
    # "720p": {1: (0.05, 36)},
    # "1024": {1: (0.05, 36)},
    # ---
    #"1080p": {1: (0.1, 5)},
    # ---
    #"2048": {1: (0.1, 5)},
    # ---
    # "360p": {1: (0.2, 141)},
    # "512": {1: (0.1, 141)},
    # # ---
    # "480p": {1: (0.1, 89)},
    # # ---
    # "720p": {1: (0.05, 36)},
    # "1024": {1: (0.05, 36)},
    # # ---
    # "1080p": {1: (0.1, 5)},
    # # ---
    # "2048": {1: (0.1, 5)},
    
    # "144p": {49: (1.0, 100)}
    # # ---
    # "360p": {1: (0.2, 5)},


    # "512": {1: (1, 5)},
    # # # ---
    # "480p": {1: (0.1, 5)},
    # # # ---
    # "720p": {1: (0.05, 5)},
    # "1024": {1: (0.5, 5)},
    # # ---
    # "1080p": {1: (0.1, 5)},
    # # ---
    # "2048": {1: (0.1, 5)},
}

grad_checkpoint = True

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
mode = "FSDP"
sp_degree = 4
monitor = True

exp_flag = "dev"
maskdit = False
unpatchify_loss = False

# Model settings
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
    maskdit=maskdit,
    decoder_depth=0,
)

# Model validate
# model = dict(
#     type="Flux-3D",
#     from_pretrained=None,
#     patch_size = 2,
#     in_channels = 64,
#     num_layers = 2,
#     num_single_layers = 4,
#     attention_head_dim = 128,
#     num_attention_heads = 24,
#     joint_attention_dim = 4096,
#     pooled_projection_dim = 768,
#     guidance_embeds = False,
#     axes_dims_rope = [16, 56, 56],
#     maskdit=maskdit,
#     decoder_depth=4,
# )

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
# vae = dict(
#     type="OpenSoraVAE_V1_2",
#     from_pretrained="/home/dufei.df/models/aigc/OpenSora-VAE-v1.2",
#     micro_frame_size=17,
#     micro_batch_size=4,
#     force_huggingface=True
# )
vae = dict(
    type="CogVideoXVAE",
    from_pretrained=pretrained_path+"/CogVideoX-2b/vae/",
)

# vae = dict(
#     type="SD3VAE",
#     from_pretrained="/home/dufei.df/models/aigc/SD3/vae/",
# )
scheduler = dict(
    type="rflow-sd3",
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
if maskdit:
    spatial_mask_ratio = 0.5
    mae_loss_coef = 0.0

prompt_uncond_prob=0.1 #for cfg
mask_ratios=None
# Log settings
seed = 42
outputs = "outputs"
wandb = False
epochs = 1000
log_every = 1
ckpt_every = 500

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-4
ema_decay = 0.99
adam_eps = 1e-15
weight_decay=1e-4
warmup_steps = 200
