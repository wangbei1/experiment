num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
seed = 42
num_eval_timesteps = 10
mode = "FSDP"

# Dataset settings
dataset = dict(
    type="VariableVideoTextWithDurationDataset",
    transform_name="resize_crop",
    sample_fps=16,
    add_one=False,
    data_path="/mnt/dufei.df/Data/hd_720p_val_local.csv",
)

# webvid
bucket_config = {
    "480p": {5: (1.0, 1)},
}

sp_degree = 8

exp_dir = "/mnt/weining.wjw/02-Codes/VidGen/outputs/044-WanX21-wanx_stage1_full_13B_nas_models_sp4_720p_5s_bs1_tonglu_H100"

text_len = 512

use_fixed_seq_len = False

# Model settings
model = dict(
    type="WanX21",
    # from_pretrained="/mnt/weining.wjw/02-Codes/VidGen/outputs/044-WanX21-wanx_stage1_full_13B_nas_models_sp4_720p_5s_bs1_tonglu_H100/epoch0-global_step400/model",
    from_pretrained=None,
    patch_size=(1, 2, 2),
    text_len=text_len,
    in_dim=16,
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
    checkpoint_path='/mnt/weining.wjw/03-Checkpoints/wanx2.1-t2v/cache/models_t5_umt5-xxl-enc-bf16.pth',
    tokenizer_path='/mnt/weining.wjw/03-Checkpoints/wanx2.1-t2v/cache/google/umt5-xxl')

vae = dict(
    type="WanX21_VAE",
    vae_pth="/mnt/weining.wjw/03-Checkpoints/wanx2.1-t2v/cache/vae.pth",)

scheduler = dict(
    type="rflow-wanx",
    num_timesteps=1000,     # num_train_timesteps
    sample_steps=50,        # sample_steps
    sample_shift=5.0,       # sample_shift
    cfg_scale=5.0,          # sample_guide_scale
    transform_scale=1,
    use_discrete_timesteps=False,
    use_timestep_transform=False,
    use_fixed_timestep_transform=False,
)