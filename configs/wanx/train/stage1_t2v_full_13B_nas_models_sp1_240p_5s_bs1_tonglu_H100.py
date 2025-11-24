_base_ = "stage1_t2v_full_13B_nas_models_sp4_720p_5s_bs1_tonglu_H100.py"

bucket_config = {
    "_delete_": True,
    "240p": {5: (1.0, 1)},
}

sp_degree = 1

model = dict(
    sp_degree=sp_degree)

exp_flag = "wanx_stage1_t2v_full_13B_nas_models_sp1_240p_5s_bs1_tonglu_H100"