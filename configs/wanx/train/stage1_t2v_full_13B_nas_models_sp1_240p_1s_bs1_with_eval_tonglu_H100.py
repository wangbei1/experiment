_base_ = "stage1_t2v_full_13B_nas_models_sp4_720p_5s_bs1_tonglu_H100.py"

eval_dataset = dict(
    type="VariableVideoTextDataset",
    data_path="/mnt/dufei.df/Data/manual_select_720p_val_96.parquet",
    transform_name="resize_crop",
)

bucket_config = {
    "_delete_": True,
    "240p": {1: (1.0, 1)},
}

eval_bucket_config = {
    "_delete_": True,
    "240p": {2: (None, 1)},
}

sp_degree = 1

model = dict(
    sp_degree=sp_degree)

exp_flag = "wanx_stage1_t2v_full_13B_nas_models_sp1_240p_5s_bs1_eval_tonglu_H100"

ckpt_every = 100
eval_every = 100