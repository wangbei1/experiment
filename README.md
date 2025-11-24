# DAMO VideoGen Codebase

## Installation

### Install Common Dependency

```bash
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install mmengine pandas timm rotary_embedding_torch ftfy accelerate diffusers av numpy tensorboard pandarallel pyarrow pre-commit openai oss2 backoff tiktoken transformers sentencepiece bs4 decord tiktoken ipdb rpdb -i http://yum.tbsite.net/pypi/simple --trusted-host yum.tbsite.net

pip install opencv-python==4.8.0.74 opencv-python-headless==4.8.0.74 -i http://yum.tbsite.net/pypi/simple --trusted-host yum.tbsite.net

pip install imageio "imageio[ffmpeg]"

# downgrade numpy
pip install numpy==1.26.3 -i https://pypi.tuna.tsinghua.edu.cn/simple

# downgrade wandb
pip install wandb==0.18.0

# downgrade av
pip install av==13.0.0

# upgrade setuptools for flash-attn installation
pip install setuptools --upgrade
```

### Install FlashAttention and Activate FA3

> Make sure that ninja is installed and that it works correctly (e.g. ninja --version then echo ? should return exit code 0). If not (sometimes ninja --version then echo $? returns a nonzero exit code), uninstall then reinstall ninja (pip uninstall -y ninja && pip install ninja). Without ninja, compiling can take a very long time (2h) since it does not use multiple CPU cores. With ninja compiling takes 3-5 minutes on a 64-core machine using CUDA toolkit.
```bash
# make sure to install ninja with proper version, like 1.11.1.3
pip uninstall -y ninja && pip install ninja==1.11.1.3

# install flash-attention 2
pip install flash-attn --no-build-isolation

# install flash-attention 3
git clone https://github.com/Dao-AILab/flash-attention.git

cd flash-attention
git submodule update --init --recursive
cd hopper
# adjust MAX_JOBS yourself
MAX_JOBS=32 python setup.py install 
```

### Install `training_acc` to activate Sequence Parallel
```bash
# Please check the package verison
pip install packages/training_acc-0.0.3-py3-none-any.whl packages/astra-0.0.1-py3-none-any.whl
```

## Config NAS (只对弹内有效)

You can mount the NASs by data config when create train/debug environments.

```bash
data_mnt_dir=/home/dufei.df/huaniu_workspace
mkdir -p $data_mnt_dir
mount -t nfs -o vers=4,nolock,noresvport 987d94b269-nhx76.cn-zhangjiakou.nas.aliyuncs.com:/ $data_mnt_dir

dufei_model_mnt_dir=/home/dufei.df/models
mkdir -p $dufei_model_mnt_dir
mount -t nfs -o vers=4,nolock,noresvport 9b37c49a33-vyb3.cn-zhangjiakou.nas.aliyuncs.com:/models $dufei_model_mnt_dir
```

## Download Models

### Download WanX Models

**Download WANX T2V moldel**

```bash
mkdir cache
wget 'https://sora-data.oss-cn-wulanchabu.aliyuncs.com/zhifan/runs/wanx_v2.1/wanx_t2v_250103/models_t5_umt5-xxl-enc-bf16.pth?OSSAccessKeyId=LTAI5tKW8HQRpo4dJ87c9czL&Expires=1772995718&Signature=IfyzhgS0HEJUusYanZbH0CKThgU%3D' -O cache/models_t5_umt5-xxl-enc-bf16.pth
wget 'https://sora-data.oss-cn-wulanchabu.aliyuncs.com/zhifan/runs/wanx_v2.1/wanx_t2v_250103/vae.pth?OSSAccessKeyId=LTAI5tKW8HQRpo4dJ87c9czL&Expires=1772995697&Signature=CKFKkKA9%2Bd8OhlbTVX8UHbdyu2c%3D' -O cache/vae.pth
wget 'https://sora-data.oss-cn-wulanchabu.aliyuncs.com/zhifan/runs/wanx_v2.1/wanx_t2v_250103/wanx_t2v_250103.pth?OSSAccessKeyId=LTAI5tKW8HQRpo4dJ87c9czL&Expires=1772995676&Signature=1VnEYuBV8KH3xCDfST6NuCiC1Bs%3D' -O cache/wanx_t2v_250103.pth
wget 'https://sora-data.oss-cn-wulanchabu.aliyuncs.com/zhifan/runs/wanx_v2.1/wanx_t2v_250103/google.tar.gz?OSSAccessKeyId=LTAI5tKW8HQRpo4dJ87c9czL&Expires=1772995624&Signature=xcMorVtwncSoVP39o7R1drPEslY%3D' -O cache/google.tar.gz

cd cache

tar -xzvf google.tar.gz
```

**Download WANX I2V model**

```bash
wget 'https://synthesis-source-se.oss-ap-southeast-1.aliyuncs.com/yupeng/shared/wanx21_i2v/cache/ema_480p.pth?OSSAccessKeyId=LTAI5t5vrnouwBwZPDMM7qV9&Expires=2054214804&Signature=Dp3n7%2BhF3PBqJ%2BVIUrR3wz2gXGY%3D' -O cache/ema_480p.pth
wget 'https://sora-data.oss-cn-wulanchabu.aliyuncs.com/yupeng/service/kf2vid_service_v2.4/ema.pth?OSSAccessKeyId=LTAI5t5vrnouwBwZPDMM7qV9&Expires=1775254957&Signature=HRtmZ1YzZW9I%2F0aQbOLyttW9VKQ%3D' -O cache/ema_720p.pth
wget 'https://synthesis-source-se.oss-ap-southeast-1.aliyuncs.com/yupeng/shared/wanx21_i2v/cache/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth?OSSAccessKeyId=LTAI5t5vrnouwBwZPDMM7qV9&Expires=2038854933&Signature=ZBnNw9DiBsNvvwz%2FP2nrEDW7zts%3D' -O cache/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
wget 'https://synthesis-source-se.oss-ap-southeast-1.aliyuncs.com/yupeng/shared/wanx21_i2v/cache/models_t5_umt5-xxl-enc-bf16.pth?OSSAccessKeyId=LTAI5t5vrnouwBwZPDMM7qV9&Expires=2038854989&Signature=Uecw%2BKcQiHrLxwHl1kyIjYBp4w8%3D' -O cache/models_t5_umt5-xxl-enc-bf16.pth
wget 'https://synthesis-source-se.oss-ap-southeast-1.aliyuncs.com/yupeng/shared/wanx21_i2v/cache/vae_step_411000.pth?OSSAccessKeyId=LTAI5t5vrnouwBwZPDMM7qV9&Expires=2038855034&Signature=ggbQqVErBncTVxNYtSs72LpFAAQ%3D' -O cache/vae_step_411000.pth
```

## Training Script Example

### Training Hunyuan

```bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD;
export TORCH_NCCL_ENABLE_MONITORING=0;
torchrun --nproc_per_node=8 scripts/HunyuanVideo/train_hunyuan.py configs/hunyuan/train/stage1_tannei.py
```

### Training Wanx2 T2V

**Training 720P 5s, SP=8:**

```bash
# Common environments
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export NCCL_NET_PLUGIN=none
# enable expandable_segments to avoid fragmentation
# Recommended use Pytorch 2.6.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use all gpus to train.
torchrun --nproc_per_node=8 scripts/WanX2.1/train_wanx2.1_t2v.py configs/wanx/train/stage1_t2v_full_13B_nas_models_sp8_720p_5s_bs1_tonglu_H100.py

# For Debug: If you want to use part of GPUS, please use CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/WanX2.1/train_wanx2.1_t2v.py configs/wanx/train/stage1_t2v_full_13B_nas_models_sp8_720p_5s_bs1_tonglu_H100.py
```

**Training 720P 5s, SP=4:**

```bash
# Common environments
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export NCCL_NET_PLUGIN=none
# enable expandable_segments to avoid fragmentation
# Recommended use Pytorch 2.6.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use all gpus to train.
torchrun --nproc_per_node=8 scripts/WanX2.1/train_wanx2.1_t2v.py configs/wanx/train/stage1_t2v_full_13B_nas_models_sp4_720p_5s_bs1_tonglu_H100.py
```

### Training Wanx2 I2V

**Train Model with SP:**

```bash
# Common environments
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export NCCL_NET_PLUGIN=none
# enable expandable_segments to avoid fragmentation
# Recommended use Pytorch 2.6.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 scripts/WanX2.1/train_wanx2.1_i2v.py configs/wanx/train/stage1_i2v_full_13B_nas_models_sp8_720p_5s_bs1_tonglu_H100.py
```

## Inference Script Example

### Hunyuan Inference

```bash
export PYTHONPATH=$PWD;
torchrun --nproc_per_node=8 scripts/HunyuanVideo/inference_hunyuan_dist.py configs/hunyuan/inference/sample.py --prompt-path assets/texts/VBench/all_category.txt --prompt-as-path --num-frames 80 --batch-size 1 --image-size 720 1280
```

### WanX T2V Inference

**Multiple GPUs With SP:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export NCCL_NET_PLUGIN=none
# enable expandable_segments to avoid fragmentation
# Recommended use Pytorch 2.6.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 scripts/WanX2.1/inference_wanx2.1_t2v_multi_gpu_sp.py configs/wanx/inference/t2v_sample_nas_model_8gpus_720p_5s_sp8_tonglu_H100.py --prompt-path assets/texts/t2v_wanx.txt --prompt-as-path
```

**Single GPUs Without SP**
```bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export NCCL_NET_PLUGIN=none
# enable expandable_segments to avoid fragmentation
# Recommended use Pytorch 2.6.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/WanX2.1/inference_wanx2.1_t2v_single_gpu.py configs/wanx/inference/t2v_sample_nas_model_1gpu_720p_5s_sp1_tonglu_H100.py --prompt-path assets/texts/t2v_wanx.txt --prompt-as-path
```

### WanX I2V Inference

**Prompt format**

Please reference `assets/texts/i2v_wanx_text_image.txt`, the reference image and prompt need to combine to a string.

```plaintext
一个女子手拎皮箱，背着各种颜色的气球向前走.{"reference_path": "assets/images/wanx2.1_i2v_official.png","mask_strategy": "0"}
```

**Multiple GPUs With SP**
```bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export NCCL_NET_PLUGIN=none
# enable expandable_segments to avoid fragmentation
# Recommended use Pytorch 2.6.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 scripts/WanX2.1/inference_wanx2.1_i2v_multi_gpu_sp.py configs/wanx/inference/i2v_sample_nas_model_8gpu_720p_5s_sp8_tonglu_H100.py --prompt-path assets/texts/i2v_wanx_text_image.txt --prompt-as-path
```

**Single GPUs Without SP:**
```bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export NCCL_NET_PLUGIN=none
# enable expandable_segments to avoid fragmentation
# Recommended use Pytorch 2.6.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/WanX2.1/inference_wanx2.1_i2v_single_gpu.py configs/wanx/inference/i2v_sample_nas_model_1gpu_720p_5s_sp1_tonglu_H100.py --prompt-path assets/texts/i2v_wanx_text_image.txt --prompt-as-path
```

### WanX V2V Inference

Please reference `assets/texts/v2v_wanx_text_video.txt`, the reference video and prompt need to combine to a string.

```plaintext
一只卡通老鼠站在一个窗户前，窗户上有蓝色的玻璃。{"reference_path": "assets/videos/杰瑞片头.mp4","mask_strategy": "0"}
```

Use `strength` parameter in config file to control noise strength，higher `strength` leads to more differences between original video and generated video.

```bash
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export NCCL_NET_PLUGIN=none
# enable expandable_segments to avoid fragmentation
# Recommended use Pytorch 2.6.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 scripts/WanX2.1/inference_wanx2.1_v2v_multi_gpu_sp.py configs/wanx/inference/v2v_sample_nas_model_8gpus_720p_5s_sp8_tonglu_H100.py --prompt-path assets/texts/v2v_wanx_text_video.txt
```

## Data

### Demo Data

- Tonglu H100:
    - Mount Data Name: `h100_nas` (NAS `038b14a77c-mhc30.cn-hangzhou.nas.aliyuncs.com`)
    - Meta Data: `/mnt/dufei.df/Data/vcg_with_image_test.parquet`
    - Videos: `oss://damo-data-hub-2/*`
- Tannei H20:
    - Mount Data Name: `huaniu_workspace` (NAS `987d94b269-nhx76.cn-zhangjiakou.nas.aliyuncs.com`)
    - Meta Data: `/home/dufei.df/huaniu_workspace/Data/test_vcg.parquet`
    - Videos: `oss://damo-data-hub-2/*`

## Debug Script Example

### Debug Wanx2

```bash
# Common environments
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$PWD
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export NCCL_NET_PLUGIN=none
# enable expandable_segments to avoid fragmentation
# Recommended use Pytorch 2.6.0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use all gpus to train.
torchrun --nproc_per_node=8 scripts/WanX2.1/train_wanx2.1_t2v.py configs/wanx/train/stage1_t2v_debug_small_3.5B_nas_models_sp8_240p_tannei_H20.py

# If you want to use part of GPUS, please use CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=8 scripts/WanX2.1/train_wanx2.1_t2v.py configs/wanx/train/stage1_t2v_debug_small_3.5B_nas_models_sp8_240p_tannei_H20.py
```

## 训练显存不足问题
遇到显存不足问题，如果是多机，可以调整train_xxx.py里的以下代码，device_num默认是8，如果是4个节点，meshi_size=(4, 8)，可以调节为(2, 16)，或者(1, 32)，根据节点数按需调整，后面的值越大，越省显存。
```
mesh_size = (_world_size // device_num, device_num)
```
