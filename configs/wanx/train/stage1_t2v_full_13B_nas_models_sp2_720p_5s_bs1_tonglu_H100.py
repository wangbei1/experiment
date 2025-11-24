_base_ = "stage1_t2v_full_13B_nas_models_sp4_720p_5s_bs1_tonglu_H100.py"

"""Training State
- State: Success (just run some steps)
- 8 * H100
- SP=8
- Batch size=1
- Latent shape: (1, 16, 20, 90, 160) -> (1, 72000, 5120)
- Need to use: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

sp_degree = 2

model = dict(
    sp_degree=sp_degree)

exp_flag = "wanx_stage1_t2v_full_13B_nas_models_sp2_720p_5s_bs1_tonglu_H100"