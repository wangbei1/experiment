_base_ = "stage1_t2v_full_13B_nas_models_sp4_720p_5s_bs1_tonglu_H100.py"

dataset = dict(
    data_path="/home/dufei.df/huaniu_workspace/Data/test_vcg.parquet",
)

sp_degree = 8

exp_flag = "wanx_stage1_t2v_full_13B_nas_models_sp8_720p_5s_bs1_tannei_H20"

# Model settings
model = dict(
    from_pretrained="/mnt/workspace/workgroup/weining.wjw/02-Codes/wanx_t2v/cache/wanx_t2v_250103.pth",
    sp_degree=sp_degree
)

t5 = dict(
    checkpoint_path='/mnt/workspace/workgroup/weining.wjw/02-Codes/wanx_t2v/cache/models_t5_umt5-xxl-enc-bf16.pth',
    tokenizer_path='/mnt/workspace/workgroup/weining.wjw/02-Codes/wanx_t2v/cache/google/umt5-xxl'
)

vae = dict(
    vae_pth="/mnt/workspace/workgroup/weining.wjw/02-Codes/wanx_t2v/cache/vae.pth"
)