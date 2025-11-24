# Dataset settings
dataset = dict(
    type="VariableVideoTextWithDurationDataset",
    transform_name="resize_crop",
    data_path="/home/dufei.df/huaniu_workspace/Data/test_vcg.parquet",
    sample_fps=16,
    add_one=False,
)

# dataset = dict(
#     type="VariableVideoTextPerRankDataset",
#     transform_name="resize_crop",
#     data_path="/mnt/dufei.df/Data/video_data_500M_filtered_240p_24s_bs4_node960/split_data_{}.parquet",
#     image_data_path="/mnt/dufei.df/Data/image_data_vcg_joy_filteredunsafe_17M7_240p_24s_bs4_node960/split_data_{}.parquet",
#     image_percent=None,
#     sample_fps=16,
#     add_one=True,
# )

bucket_config = {
    "240p": {2: (1.0, 6), 4: (1.0, 4)},
}

grad_checkpoint = True

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
mode = "FSDP"
sp_degree = 4
monitor = False

exp_flag = "hunyuan_test"
maskdit = False
unpatchify_loss = False

# Model settings
model = dict(
    type="Hunyuan",
    # from_pretrained="/home/dufei.df/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/transformers/model.safetensors",
    from_pretrained=None,
    text_states_dim=4096,
    text_states_dim_2=768,
    patch_size=[1, 2, 2],
    in_channels=16,  # Should be VAE.config.latent_channels.
    out_channels=None,
    hidden_size=3072,
    heads_num=24,
    mlp_width_ratio=4.0,
    mlp_act_type="gelu_tanh",
    mm_double_blocks_depth=20,
    mm_single_blocks_depth=40,
    rope_dim_list=[16, 56, 56],
    qkv_bias=True,
    qk_norm=True,
    qk_norm_type="rms",
    guidance_embed=True,  # For modulation.
    text_projection="single_refiner",
    use_attention_mask=True,
)

text_encoder=dict(
    text_encoder_path="/home/dufei.df/models/aigc/hunyuan_ckpts/text_encoder/",
    text_encoder_type="llm",
    max_length=256,
    tokenizer_type="llm",
    hidden_state_skip_layer=2,
)
text_encoder_2=dict(
    text_encoder_path="/home/dufei.df/models/aigc/hunyuan_ckpts/text_encoder_2/",
    text_encoder_type="clipL",
    max_length=77,
    tokenizer_type="clipL",
)

vae = dict(
    type="Hunyuan_VAE",
    from_pretrained="/home/dufei.df/models/aigc/hunyuan_ckpts/hunyuan-video-t2v-720p/vae/",
)

scheduler = dict(
    type="rflow-sd3",
    use_timestep_transform=True,
    use_fixed_timestep_transform=True,
    sample_method="logit-normal",
    transform_scale=9,
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
outputs = "outputs/"
wandb = False
epochs = 1
log_every = 1
ckpt_every = 500
# eval_every = 200

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-5
ema_decay = 0.99
adam_eps = 1e-15
weight_decay= 1e-4
warmup_steps = 200