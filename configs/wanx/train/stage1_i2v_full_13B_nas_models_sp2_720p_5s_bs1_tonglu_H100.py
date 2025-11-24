
# Dataset settings
dataset = dict(
    type="VariableVideoTextWithDurationDataset",
    transform_name="resize_crop",
    data_path="/mnt/dufei.df/Data/vcg_with_image_test.parquet",
    sample_fps=16,
    add_one=True,
)

bucket_config = {
    "720p": {5: (1.0, 1)},
}

grad_checkpoint = True
text_len = 512

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
mode = "FSDP"
sp_degree = 2
monitor = False

exp_flag = "wanx_stage1_i2v_full_13B_nas_models_sp2_720p_5s_bs1_tonglu_H100"
maskdit = False
unpatchify_loss = False

use_fixed_seq_len = False

in_dim = 2 * 16 + 4

# Model settings
model = dict(
    type="WanX21_I2V",
    from_pretrained="/mnt/weining.wjw/02-Codes/wanx2.1_i2v/cache/ema_720p.pth",
    # from_pretrained=None, # for debug
    patch_size=(1, 2, 2),
    text_len=text_len,
    in_dim=in_dim,
    dim=5120,
    ffn_dim=13824,
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=40,
    num_layers=40,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
    use_fixed_seq_len=use_fixed_seq_len,
    sp_degree=sp_degree)

t5 = dict(
    name='umt5_xxl',
    text_len=text_len,
    dtype=dtype,
    checkpoint_path='/mnt/weining.wjw/02-Codes/wanx2.1_i2v/cache/models_t5_umt5-xxl-enc-bf16.pth',
    tokenizer_path='/mnt/weining.wjw/02-Codes/wanx2.1_i2v/cache/umt5-xxl')

vae = dict(
    type="WanX21_VAE",
    vae_pth="/mnt/weining.wjw/02-Codes/wanx2.1_i2v/cache/vae_step_411000.pth")


clip_model = 'clip_xlm_roberta_vit_h_14'
clip_checkpoint = '/mnt/weining.wjw/02-Codes/wanx2.1_i2v/cache/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
clip_tokenizer = '/mnt/weining.wjw/02-Codes/wanx2.1_i2v/cache/xlm-roberta-large'

max_seq_len = 75600

sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

scheduler = dict(
    type="rflow-wanx",
    num_timesteps=1000,     # num_train_timesteps
    sample_steps=40,
    sample_shift=5.0,       # sample_shift
    cfg_scale=5.0,          # sample_guide_scale
    transform_scale=5.0,
    use_discrete_timesteps=False,
    sample_method="logit-normal",
    use_timestep_transform=True,
    use_fixed_timestep_transform=True)

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
ckpt_every = 200
# eval_every = 100

# optimization settings
load = "latest"
grad_clip = 1.0
lr = 1e-5
ema_decay = 0.99
adam_eps = 1e-15
weight_decay= 1e-4
warmup_steps = 200