"""Training State
- State: Success (just run some steps)
- 8 * H100
- SP=4
- Batch size=1
- Latent shape: (1, 16, 20, 90, 160) -> (1, 72000, 5120)
- Use 78GB memory
- 30s/it
- Need to use: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Dataset settings
# qinghai H100: data_path = "/mnt/groupnas_zjk/weining.wjw/02-Data/test_vcg_nas_10000.parquet"
# tonglu H100: data_path = "/mnt/dufei.df/Data/vcg_with_image_test.parquet"
dataset = dict(
    type="VariableVideoFlowTextWithDurationDataset",
    transform_name="resize_crop",
    data_path="/data/wubin/data_mixkit/data/mixkit_480p_5s.parquet",
    sample_fps=16,
    add_one=True,
    video_root="/data/wubin/data_mixkit/data/video",
    flow_root="/data/wubin/data_mixkit/data/flow",
)

bucket_config = {
    # "_delete_": True,
    "720p": {5: (1.0, 1)},
}

grad_checkpoint = True
text_len = 512

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
mode = "FSDP"
sp_degree = 4
monitor = False

exp_flag = "wanx_stage1_t2v_full_13B_nas_models_sp4_720p_5s_bs1_tonglu_H100"
maskdit = False
unpatchify_loss = False

# Model settings
model = dict(
    type="WanX21",
    from_pretrained="/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors", # TODO.
    # from_pretrained=None, # for debug
    patch_size=(1, 2, 2),
    text_len=text_len,
    in_dim=16,
    dim=1536, # 5120->1536
    ffn_dim=8960, # 13824->8960
    freq_dim=256,
    text_dim=4096,
    out_dim=16,
    num_heads=12, # 40->12
    num_layers=30, # 40->30
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
    use_fixed_seq_len=False,
    sp_degree=sp_degree)

t5 = dict(
    name='umt5_xxl',
    text_len=text_len,
    dtype=dtype,
    checkpoint_path='/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth', # TODO.
    tokenizer_path='/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl') # TODO.

vae = dict(
    type="WanX21_VAE",
    vae_pth="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE_for_wanx_code.pth",) # TODO.

max_seq_len = 75600

sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

scheduler = dict(
    type="rflow-wanx",
    num_timesteps=1000,     # num_train_timesteps
    sample_steps=50,
    sample_shift=5.0,       # sample_shift
    cfg_scale=5.0,          # sample_guide_scale
    transform_scale=5.0,
    use_discrete_timesteps=False,
    sample_method="logit-normal",
    use_timestep_transform=True,
    use_fixed_timestep_transform=True,
    flow_loss_weight=1.0,   # 以后可以改成 2.0, 5.0 等
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
epochs = 100
log_every = 1
ckpt_every = 100
# eval_every = 100

# optimization settings
grad_clip = 1.0
lr = 1e-5
ema_decay = 0.99
adam_eps = 1e-15
weight_decay= 1e-4
warmup_steps = 200



sample_every = 2000  # 每 2000 个 global_step 采一次（自己调）
sample_fps = 16       # 输出视频的 fps
dtype= "bf16"            # 或 fp16
fps= 16
num_frames= "4s"       # 或者你 inference 用的那种写法（和 get_num_frames 对应）
resolution= "480p"
aspect_ratio= "9:16"


# sample_prompt= "A stylish woman ..."   # 可选，不写就用我在代码里的默认
# sample_neg_prompt= ""  # 可选，空字符串也没问题


