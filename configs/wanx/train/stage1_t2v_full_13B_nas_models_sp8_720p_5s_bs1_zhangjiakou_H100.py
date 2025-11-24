_base_ = "stage1_t2v_full_13B_nas_models_sp4_720p_5s_bs1_tonglu_H100.py"

# Dataset settings
# qinghai H100: data_path = "/mnt/groupnas_zjk/weining.wjw/02-Data/test_vcg_nas_10000.parquet"
# tonglu H100: data_path = "/mnt/dufei.df/Data/vcg_with_image_test.parquet"
dataset = dict(
    data_path="/mnt/groupnas_zjk/weining.wjw/02-Data/test_vcg_nas_10000.parquet",
)

sp_degree = 8

exp_flag = "wanx_stage1_t2v_full_13B_nas_models_sp8_720p_5s_bs1_zhangjiakou_H100"

# Model settings
model = dict(
    from_pretrained="/mnt/groupnas_zjk/weining.wjw/03-Weights/wanx/cache/wanx_t2v_250103.pth",
    sp_degree=sp_degree
)

t5 = dict(
    checkpoint_path='/mnt/groupnas_zjk/weining.wjw/03-Weights/wanx/cache/models_t5_umt5-xxl-enc-bf16.pth',
    tokenizer_path='/mnt/groupnas_zjk/weining.wjw/03-Weights/wanx/cache/google/umt5-xxl'
)

vae = dict(
    vae_pth="/mnt/groupnas_zjk/weining.wjw/03-Weights/wanx/cache/vae.pth"
)
