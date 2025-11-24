resolution = "360p"
aspect_ratio = "9:16"
num_frames = 49
fps = 16
frame_interval = 1
save_fps = 16

save_dir = "./samples/samples/"
seed = 0
batch_size = 1
multi_resolution = "OpenSora"
dtype = "bf16"
condition_frame_length = 5
align = None
sp_degree = 1
save_dir = "Eval_samples_webvidval/hunyuan_moyuan_017_maskdit_fulltoken_480p_5s_step21k_inferencestep50"

model = dict(
    type="Hunyuan",
    from_pretrained="/mnt/moyuan.yty/Outputs/017-Hunyuan-maskdit_fulltoken_480p_5s/epoch0-global_step21000/",
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
    guidance_embed=False,  # For modulation.
    text_projection="single_refiner",
    use_attention_mask=True,
)

text_encoder=dict(
    text_encoder_path="/mnt/dufei.df/HunyuanVideo/ckpts/text_encoder/",
    text_encoder_type="llm",
    max_length=256,
    tokenizer_type="llm",
    hidden_state_skip_layer=2,
)
text_encoder_2=dict(
    text_encoder_path="/mnt/dufei.df/HunyuanVideo/ckpts/text_encoder_2/",
    text_encoder_type="clipL",
    max_length=77,
    tokenizer_type="clipL",
)

vae = dict(
    type="JINGKAI_VAE",
    pretrained_spatial_vae_path="/mnt/dufei.df/FLUX.1-dev-diffusers/",
    from_pretrained="/mnt/dufei.df/video_vae/checkpoint-stage4-final.pth",
    scaling_factor=0.9480883479118347,
    shift_factor=0.04052285850048065,
)

scheduler = dict(
    type="rflow-sd3",
    use_timestep_transform=True,
    use_fixed_timestep_transform=True,
    num_sampling_steps=50,
    cfg_scale=4.5,
    transform_scale=9,
)

aes = None
flow = None