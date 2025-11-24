_base_ = "stage1_t2v_full_13B_nas_models_sp4_720p_5s_bs1_tonglu_H100.py"

"""Training State
- State: Train Failed!!!
- 8 * H100
- SP=4
- Batch size=2
- Latent shape: (2, 16, 20, 90, 160) -> (2, 72000, 5120)
"""

bucket_config = {
    "_delete_": True,
    "720p": {5: (1.0, 2)},
}

exp_flag = "wanx_stage1_t2v_full_13B_nas_models_sp8_720p_5s_bs2_tonglu_H20"