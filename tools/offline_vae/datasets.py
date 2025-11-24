
import os
from glob import glob
import json
import math
import numpy as np
import torch
from PIL import ImageFile
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
import pandas as pd
from copy import deepcopy
from vidgen.datasets.utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop
from vidgen.datasets.fs import read_pil_image, read
import random

import decord
decord.bridge.set_bridge("torch")

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_FPS = 120

class VariableVideoTextPerRankDatasetVae(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path=None,
        image_data_path=None,
        rank=None,
        num_frames=None,
        frame_interval=1,
        image_size=(None, None),
        transform_name=None,
        dummy_text_feature=False,
        sample_fps = 16,
        image_percent = None,
        add_one=False,
        **kwargs
    ):
        self.rank = rank
        assert data_path is not None
        data = read_file(data_path.format(int(rank)))
        if image_data_path is not None:
            image_data = read_file(image_data_path.format(int(rank)))
            self.data = pd.concat([data, image_data]).reset_index(drop=True)
        else:
            self.data = data
        self.bucket_id_list = sorted(self.data['bucket_id'].drop_duplicates().to_list())
        self.bucket_id_num_count = self.data['bucket_id'].value_counts()
        self.image_percent = image_percent
        self.total_lens = len(self.data)
        
        # self.data = self.data[self.data['path'].str.endswith(".mp4")][:100]
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }
        self.transform_name = transform_name
        # self.data['height'] = self.data['height'] * 4
        # self.data['width'] = self.data['width'] * 4
        self.dummy_text_feature = dummy_text_feature
        self.sample_fps = sample_fps
        self.add_one = add_one
    
    def __len__(self):
        return self.total_lens
    
    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"
        
    def getitem(self, index):        
        sample = deepcopy(self.data.iloc[index])
        
        bucket_id, path = sample['bucket_id'], sample['path']
        
        duration, height, width, _ = [int(val) for val in bucket_id.split("-")]
        
        path = path.replace("oss://", "/root/")
        file_type = self.get_type(path)
        ar = height / width
        
        if file_type == "video":
            video_fp = read(path)
            video_reader = decord.VideoReader(video_fp)
            ori_fps = video_reader.get_avg_fps()
            ori_video_length = len(video_reader)
            
            num_frames = int(duration * self.sample_fps)
            if self.add_one:
                num_frames = num_frames + 1
            required_len = math.ceil(num_frames / self.sample_fps * ori_fps)
            
            clip_len = min(10, max(ori_video_length - required_len, 0) // 2)
            
            normed_video_length = round((ori_video_length - 2*clip_len) / ori_fps * self.sample_fps)
            
            normed_video_length = max(num_frames, normed_video_length)
            
            batch_index_all = np.linspace(clip_len, ori_video_length - 1 - clip_len, normed_video_length).round().astype(int)
            start_idx = 0 #random.randint(0, normed_video_length - num_frames)
            batch_index = batch_index_all[start_idx:start_idx+num_frames]
            
            video = video_reader.get_batch(batch_index).permute(0, 3, 1, 2)

            video_fps = self.sample_fps

            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
        else:
            # loading
            if path.startswith('oss'):
                image = read_pil_image(path)
            else:
                image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)
            num_frames = 1
            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        
        path = path.replace("/root/damo-data-hub-2/", "/root/damo-data-hub-2/261867/vae_latents/")
        
        path = os.path.splitext(path)[0]
        
        ret = {
            "video": video,
            "path": path,
        }
        
        return ret

    def __getitem__(self, index): 
        while True:
            try:
                return self.getitem(index)
            except Exception as e:
                print("--------------------------------- read data error: ", e)
                sample = deepcopy(self.data.iloc[index])
                bucket_id = sample['bucket_id']
                duration, height, width, _ = [int(val) for val in bucket_id.split("-")]        

                if self.bucket_id_num_count[bucket_id] == 1:
                    candidate_bucket_id = []
                    for k in self.bucket_id_list:
                        sub_duration = int(k.split("-")[0])
                        if duration == 0 and sub_duration != 0:
                            continue
                        elif subduration < duration:
                            continue
                        candidate_bucket_id.append(k)
                    bucket_id = np.random.choice(candidate_bucket_id)
                index =  np.random.choice(self.data.index[self.data['bucket_id'] == bucket_id].tolist())