_base_ = "stage1_t2v_full_13B_nas_models_sp4_720p_5s_bs1_tonglu_H100.py"

# Dataset settings
# qinghai H100: data_path = "/mnt/groupnas_zjk/weining.wjw/02-Data/test_vcg_nas_10000.parquet"
# tonglu H100: data_path = "/mnt/dufei.df/Data/vcg_with_image_test.parquet"

sp_degree = 8

exp_flag = "wanx_stage1_t2v_full_13B_nas_models_sp8_seq75600_bs1_tonglu_H100"

# Model settings
model = dict(
    use_fixed_seq_len=True,
    sp_degree=sp_degree)

max_seq_len = 75600