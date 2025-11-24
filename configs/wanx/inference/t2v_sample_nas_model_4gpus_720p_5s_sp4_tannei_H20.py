_base_ = "t2v_sample_nas_model_8gpus_720p_5s_sp8_tonglu_H100.py"

save_dir = "./samples/wanx_sample_nas_model_8gpus_720p_5s_sp4_tannei_H20/"


sp_degree = 4

use_fixed_seq_len = True

model = dict(
    type="WanX21_mixfp32",
    from_pretrained="/mnt/workspace/workgroup/weining.wjw/02-Codes/wanx_t2v/cache/wanx_t2v_250103.pth",
    use_fixed_seq_len=use_fixed_seq_len,
    sp_degree=sp_degree
)

t5 = dict(
    checkpoint_path='/mnt/workspace/workgroup/weining.wjw/02-Codes/wanx_t2v/cache/models_t5_umt5-xxl-enc-bf16.pth',
    tokenizer_path='/mnt/workspace/workgroup/weining.wjw/02-Codes/wanx_t2v/cache/google/umt5-xxl'
)

vae = dict(
    vae_pth="/mnt/workspace/workgroup/weining.wjw/02-Codes/wanx_t2v/cache/vae.pth"
)


prompt_as_path = False