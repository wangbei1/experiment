resolution = "360p"
aspect_ratio = "9:16"
num_frames = 49
fps = 25
frame_interval = 1
save_fps = 25

seed = 0
batch_size = 1
multi_resolution = "OpenSora"
dtype = "bf16"
condition_frame_length = 5
align = None

save_dir = "samples"
verbose = 2

model = dict(
    type="Hunyuan",
    from_pretrained="/home/dufei.df/models/aigc/hunyuan_ckpts/hunyuan-video-t2v-720p/transformers/model.safetensors",
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
    num_sampling_steps=30,
    cfg_scale=4.5,
    transform_scale=9,
)

aes = None
flow = None