_base_ = 't2v_sample_nas_model_1gpu_720p_5s_sp1_tonglu_H100.py'  # 引用基础配置

save_dir = "./samples/wanx_sample_local_model_1gpu_720p_5s_sp1_tonglu_H100/"

model = dict(
    from_pretrained="/root/.cache/checkpoints/wanx/cache/wanx_t2v_250103.pth")

t5 = dict(
    checkpoint_path='/root/.cache/checkpoints/wanx/cache/models_t5_umt5-xxl-enc-bf16.pth',
    tokenizer_path='/root/.cache/checkpoints/wanx/cache/google/umt5-xxl')

vae = dict(
    vae_pth="/root/.cache/checkpoints/wanx/cache/vae.pth")