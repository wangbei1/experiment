_base_ = "stage1_t2v_full_13B_nas_models_sp4_720p_5s_bs1_tonglu_H100.py"

"""Training State
- State: Success (run some steps)
- 8 * H100
- SP=1
- Batch size=1
# - Latent shape: (1, 16, 20, 60, 106) -> (1, 31800, 5120)
# - Use 78GB memory
# - 36s/it
# - Need to use: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

bucket_config = {
    "_delete_": True,
    "480p": {5: (1.0, 2)},
}

sp_degree = 1

model = dict(
    sp_degree=sp_degree)


# 是否使用双分支结构

# from_pretrained 是哪种风格的 ckpt:
#   False: 原生 16 通道 Wan 预训练
#   True:  已经扩过的 32 通道 VideoJAM 模型
pretrained_is_videojam = False

# 单分支 latent 通道数（= VAE 的 z_dim）
base_latent_channels = 16

exp_flag = "videojam"
text_len = 512

model = dict(
    type="WanX21",
    from_pretrained="/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model_videojam_32ch_dualhead.safetensors",
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=32,
    dim=1536,
    ffn_dim=8960,
    freq_dim=256,
    text_dim=4096,
    out_dim=32,
    num_heads=12,
    num_layers=30,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    eps=1e-6,
    use_fixed_seq_len=False,
    sp_degree=sp_degree,
    use_dual_head=True,          # ⭐ 记得打开
)

enable_videojam = True
log_every = 1
ckpt_every = 100
outputs = "outputs_videojam/two_heads/"
debug_videojam_loss=False



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
    enable_videojam=True,   
    video_loss_weight=1.0,   # ⭐ 先只训 flow
    debug_videojam_loss=False,
)
