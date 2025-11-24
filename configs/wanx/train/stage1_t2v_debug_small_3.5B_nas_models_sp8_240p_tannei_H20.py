_base_ = "stage1_t2v_full_13B_nas_models_sp4_720p_5s_bs1_tonglu_H100.py"
# Dataset settings
dataset = dict(
    data_path="/home/dufei.df/huaniu_workspace/Data/test_vcg.parquet",
    sample_fps=4,
)

bucket_config = {
    "_delete_": True,
    "240p": {2: (1.0, 1), 4: (1.0, 1)},
}

sp_degree = 8

exp_flag = "wanx_stage1_t2v_small_3.5B_nas_models_sp4_720p_5s_bs2_tonglu_H100"

# Model settings
model = dict(
    # from_pretrained="/mnt/workspace/workgroup/weining.wjw/02-Codes/wanx_t2v/cache/wanx_t2v_250103.pth",
    from_pretrained=None, # for debug
    num_layers=10,  # for debug, original num is 40
    use_fixed_seq_len=False,
    sp_degree=sp_degree)

t5 = dict(
    checkpoint_path='/mnt/workspace/workgroup/weining.wjw/02-Codes/wanx_t2v/cache/models_t5_umt5-xxl-enc-bf16.pth',
    tokenizer_path='/mnt/workspace/workgroup/weining.wjw/02-Codes/wanx_t2v/cache/google/umt5-xxl')

vae = dict(
    vae_pth="/mnt/workspace/workgroup/weining.wjw/02-Codes/wanx_t2v/cache/vae.pth")
