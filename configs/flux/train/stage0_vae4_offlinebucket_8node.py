# Dataset settings
dataset = dict(
    type="VariableVideoTextPerRankDataset",
    transform_name="resize_crop",
    data_path="/home/dufei.df/huaniu_workspace/Data/image_data_vcg_joy_smalltest_240p_bs20_node8/split_data_{}.parquet",
    sample_fps=16,
)

eval_dataset = dict(
    type="VariableVideoTextWithDurationDataset",
    data_path="/home/dufei.df/huaniu_workspace/Data/cosmopolitan_eval_dataset.parquet",
    transform_name="resize_crop",
)

eval_bucket_config = {
    "512": {0: (None, 1)},
}

grad_checkpoint = True

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
mode = "FSDP"
sp_degree = 1
monitor = False

maskdit = False
unpatchify_loss = False
exp_flag = "test"#"maskdit-fulltoken"
# Model settings
model = dict(
    type="Flux-3D",
    # from_pretrained="/home/dufei.df/models/aigc/FLUX-mt5-xxl/transformer/model.safetensors",
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
    t5_embedder=True,
    maskdit=maskdit,
    decoder_depth=0,
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

scheduler = dict(
    type="rflow-sd3",
    use_timestep_transform=True,
    sample_method="logit-normal",
)

# Mask settings
# mask_ratios = {
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
eval_every = 200

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-4
ema_decay = 0.99
adam_eps = 1e-8
weight_decay=1e-4
warmup_steps = 200