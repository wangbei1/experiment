import os
import math
import torch
import numpy as np
import video_transforms
from bucket import SecondsBucket
import torchvision.transforms as transforms
from jingkai_vae import JINGKAI_VAE
import decord
decord.bridge.set_bridge("torch")


def get_video_tensor(bucket, video_path, duration, height, width):
    bucket_id = bucket.get_bucket_id(duration, height, width)
    real_t, real_h, real_w = bucket.get_thw(bucket_id)
    
    print(real_t, ' ', real_h, ' ', real_w)

    video_reader = decord.VideoReader(video_path)

    ori_fps = video_reader.get_avg_fps()
    ori_video_length = len(video_reader)

    num_frames = int(real_t * sample_fps)
    required_len = math.ceil(num_frames / sample_fps * ori_fps)

    clip_len = min(10, max(ori_video_length - required_len, 0) // 2)

    normed_video_length = round((ori_video_length - 2*clip_len) / ori_fps * sample_fps)

    normed_video_length = max(num_frames, normed_video_length)

    batch_index_all = np.linspace(clip_len, ori_video_length - 1 - clip_len, normed_video_length).round().astype(int)
    start_idx = 0 #random.randint(0, normed_video_length - num_frames)
    batch_index = batch_index_all[start_idx:start_idx+num_frames]

    video = video_reader.get_batch(batch_index).permute(0, 3, 1, 2)

    #resize_crop
    transform_video = transforms.Compose(
        [
            video_transforms.ToTensorVideo(),  # TCHW
            video_transforms.ResizeCrop((real_h, real_w)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    # transform
    video = transform_video(video)  # T C H W

    video = video.permute(1, 0, 2, 3)[None] #B C T H W
    return video

def encode_video(vae, x):
    with torch.no_grad():
        h, w = x.shape[-2:]
        batch_slice_2d = max(2 ** 22 // h // w, 1)
        z = vae.encode(x, time_slice=32, batch_slice_2d=batch_slice_2d).latent_dist.parameters
    return z

if __name__ == "__main__":
    spatial_vae_path = "/home/dufei.df/models/aigc/FLUX.1-dev-diffusers/"
    vae_path = "/home/dufei.df/models/aigc/video_vae/checkpoint-stage4-final.pth"
    bucket_config = {
        "720p": {6: (1.0, 1), 8: (1.0, 1), 10: (1.0, 1)},
    }
    sample_fps = 16
    bucket = SecondsBucket(bucket_config)
    
    vae_model = JINGKAI_VAE(
        pretrained_spatial_vae_path=spatial_vae_path,
        from_pretrained=vae_path,
        scaling_factor=0.9480883479118347,
        shift_factor=0.04052285850048065
    ).to("cuda", dtype=torch.bfloat16)
    
    vae_model.eval()
    
    video_path = "video_gen1722913899758.mp4"
    #这个信息从meta中获取的，也可以直接从视频中获取
    duration, height, width = 6, 960, 1440
    
    video = get_video_tensor(bucket, video_path, duration, height, width)
    video = video.to("cuda", dtype=torch.bfloat16)
    res = encode_video(vae_model, video).squeeze(0)
    print(res.shape)
    