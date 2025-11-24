# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop"
)

# webvid
bucket_config = {  # 12s/it
    "512": {1: (1, 72), 49: (1.0, 3), 97: ((1.0, 0.33), 1)},
    # ---
    "480p": {1: (0.5, 48), 49: (0.5, 2), 97: ((0.5, 0.33), 1)},
    # ---
    #"720p": {1: (0.3, 24), 48: (0.3, 2), 96: ((0.3, 0.33), 1)},
}

grad_checkpoint = True

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
mode = "FSDP"
sp_degree = 1
monitor = False

exp_flag = "dev-jingkaivae"
maskdit = False
fulltoken = False
unpatchify_loss = False

# Model settings
pretrained_path = "/mnt/workspace/workgroup/moyuan.yty/.cache/flux"
#pretrained_path+"/FLUX.1-dev-diffusers/transformer_onefile_bfp16/diffusion_pytorch_model.bfp16.safetensors"
model = dict(
    type="Flux-3D",
    from_pretrained=pretrained_path+"/FLUX.1-dev-diffusers/transformer_onefile_bfp16/diffusion_pytorch_model.bfp16.safetensors",#"/mnt/nas_szy/aigc/VidGen/outputs/032-Flux-3D/epoch0-global_step2000/ema",#pretrained_path+"/032-Flux-3D/epoch0-global_step6000/ema",
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
    decoder_depth=4,
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
    type="JINGKAI_VAE",
    pretrained_spatial_vae_path=pretrained_path+"/FLUX.1-dev-diffusers/",
    from_pretrained=pretrained_path+"/jingkai_vae/checkpoint-stage3-final.pth",
)


scheduler = dict(
    type="rflow-sd3",
    use_timestep_transform=True,
    sample_method="logit-normal",
)

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

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-4
ema_decay = 0.99
adam_eps = 1e-15
weight_decay=1e-4
warmup_steps = 200
