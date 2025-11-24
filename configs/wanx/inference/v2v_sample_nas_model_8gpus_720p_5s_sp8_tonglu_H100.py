image_size = (1280, 720)
aspect_ratio = "9:16"
num_frames = 81
fps = 16
frame_interval = 1
save_fps = 16

strength = 0.95

save_dir = f"./samples/wanx2.1_video2video_strength{strength}/"
seed = 0
batch_size = 1
multi_resolution = "OpenSora"
dtype = "bf16"
condition_frame_length = 5
align = None

sp_degree = 8

text_len = 512

mode = "FSDP"

max_seq_len = 75600
use_fixed_seq_len = True

model = dict(
    type="WanX21",
    from_pretrained="/mnt/weining.wjw/03-Checkpoints/wanx2.1-t2v/cache/wanx_t2v_250103.pth",
    # from_pretrained=None,
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

sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

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

aes = None
flow = None

prompt_as_path = False