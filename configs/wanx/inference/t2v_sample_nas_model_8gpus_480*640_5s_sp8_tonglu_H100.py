_base_ = 't2v_sample_nas_model_8gpus_720p_5s_sp8_tonglu_H100.py'  # 引用基础配置

resolution = "480p"
aspect_ratio = "3:4"
sp_degree = 8

save_dir = "./samples/wanx_sample_nas_model_8gpus_480_640_5s_sp8_tonglu_H100/"

max_seq_len = 25200

model = dict(
    use_fixed_seq_len=False,
    sp_degree=sp_degree)