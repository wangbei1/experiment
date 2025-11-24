_base_ = "stage1_t2v_full_13B_nas_models_sp4_720p_5s_bs1_tonglu_H100.py"

"""Training State
- State: Success (just run some steps)
- 8 * H100
- SP=8
- Batch size=1
- Latent shape: (1, 16, 40, 90, 160) -> (1, 14400, 5120)
- Use 70GB memory
- 45s/it
- Need to use: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""
# Dataset settings
# qinghai H100: data_path = "/mnt/groupnas_zjk/weining.wjw/02-Data/test_vcg_nas_10000.parquet"
# tonglu H100: data_path = "/mnt/dufei.df/Data/vcg_with_image_test.parquet"

bucket_config = {
    "_delete_": True,
    "720p": {10: (1.0, 1)},
}

sp_degree = 8

model = dict(
    sp_degree=sp_degree)

exp_flag = "wanx_stage1_t2v_full_13B_nas_models_sp8_720p_10s_bs1_tonglu_H100"

max_seq_len = 75600 * 2