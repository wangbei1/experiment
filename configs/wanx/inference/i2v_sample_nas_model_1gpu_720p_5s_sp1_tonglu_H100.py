_base_ = "i2v_sample_nas_model_8gpu_720p_5s_sp8_tonglu_H100.py"

save_dir = "./samples/wanx_i2v_sample_nas_model_1gpu_720p_5s_sp1_tonglu_H100/"

sp_degree = 1

model = dict(
    sp_degree=sp_degree)
