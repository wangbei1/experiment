import os
from glob import glob
import json
import math
import numpy as np
import torch
from PIL import ImageFile
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
import pandas as pd
from vidgen.registry import DATASETS
from copy import deepcopy
from .read_video import read_video
from .utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop
from .fs import read_pil_image, read
import random

import decord
decord.bridge.set_bridge("torch")

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_FPS = 120


@DATASETS.register_module()
class VideoTextDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_path=None,
        num_frames=16,
        frame_interval=1,
        image_size=(256, 256),
        transform_name="center",
    ):
        self.data_path = data_path
        self.data = read_file(data_path)
        # self.data = self.data[self.data['path'].str.endswith(".mp4")][:100]
        self.get_text = "text" in self.data.columns
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }

    def _print_data_number(self):
        num_videos = 0
        num_images = 0
        for path in self.data["path"]:
            if self.get_type(path) == "video":
                num_videos += 1
            else:
                num_images += 1
        print(f"Dataset contains {num_videos} videos and {num_images} images.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        file_type = self.get_type(path)

        if file_type == "video":
            # loading
            vframes, vinfo = read_video(path, backend="av")
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # Sampling video frames
            video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)

            # transform
            transform = self.transforms["video"]
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = self.transforms["image"]
            image = transform(image)

            # repeat
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        ret = {"video": video, "fps": video_fps}
        if self.get_text:
            ret["text"] = sample["text"]
        return ret

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                path = self.data.iloc[index]["path"]
                print(f"data {path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


@DATASETS.register_module()
class VariableVideoTextDataset(VideoTextDataset):
    def __init__(
        self,
        data_path=None,
        num_frames=None,
        frame_interval=1,
        image_size=(None, None),
        transform_name=None,
        dummy_text_feature=False,
        **kwargs
    ):
        super().__init__(data_path, num_frames, frame_interval, image_size, transform_name=None)
        self.transform_name = transform_name
        self.data["id"] = np.arange(len(self.data))
        # self.data['height'] = self.data['height'] * 4
        # self.data['width'] = self.data['width'] * 4
        self.dummy_text_feature = dummy_text_feature

    def get_data_info(self, index):
        T = self.data.iloc[index]["num_frames"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]
        return T, H, W

    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        index, num_frames, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        path = sample["path"].replace("/mnt/sh_nas/moyuan.yty/", "/home/dufei.df/huaniu_workspace/")
        file_type = self.get_type(path)
        ar = height / width

        # sample_fps = 24  # default fps
        if file_type == "video":
            # loading
            # vframes, vinfo = read_video(path, backend="av")
            # video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # # Sampling video frames
            # video = temporal_random_crop(vframes, num_frames, self.frame_interval)
            
            # video = video.clone()
            # del vframes
            video_fp = read(path)
            video_reader = decord.VideoReader(video_fp)
            video_fps = video_reader.get_avg_fps()
            ori_video_length = len(video_reader)
            # normed_video_length = round(ori_video_length / ori_fps * self.sample_fps)
            
            start_idx = np.random.randint(0, ori_video_length - num_frames * self.frame_interval)
            batch = np.arange(start_idx, start_idx + num_frames * self.frame_interval, self.frame_interval)
            video = video_reader.get_batch(batch).permute(0, 3, 1, 2)

            video_fps = video_fps // self.frame_interval

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

            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        ret = {
            "video": video,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
        }
        if self.get_text:
            ret["text"] = sample["text"]
        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len
        return ret

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except Exception as e:
            print("--------------------------------- read data error: ", e)
            return None


@DATASETS.register_module()
class VariableVideoTextWithDurationDataset(VideoTextDataset):
    def __init__(
        self,
        data_path=None,
        lmdb_path=None,
        num_frames=None,
        frame_interval=1,
        image_size=(None, None),
        transform_name=None,
        dummy_text_feature=False,
        sample_fps = 16,
        add_one=False,
        **kwargs
    ):
        super().__init__(data_path, num_frames, frame_interval, image_size, transform_name=None)
        self.transform_name = transform_name
        self.data["id"] = np.arange(len(self.data))
        # self.data['height'] = self.data['height'] * 4
        # self.data['width'] = self.data['width'] * 4
        self.dummy_text_feature = dummy_text_feature
        self.sample_fps = sample_fps
        self.add_one = add_one
        if lmdb_path is not None:
            self.env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
            # self.txn = self.env.begin(buffers=True, write=False)

    def get_data_info(self, index):
        T = self.data.iloc[index]["duration"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]
        return T, H, W

    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        index, duration, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        if 'path' not in sample:
            key = sample["key"]
            with self.env.begin(write=False) as txn:
                info = pickle.loads(txn.get(key.encode("ascii")))
            path, text = info
        else:
            path = sample["path"].replace("/mnt/sh_nas/moyuan.yty/", "/home/dufei.df/huaniu_workspace/")
        # path = path.replace("oss://", "/root/")
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
        ret = {
            "video": video,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
        }
        if self.get_text:
            ret["text"] = sample["text"]
        else:
            ret["text"] = text
        
        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len
        return ret

    def __getitem__(self, index):
        from training_acc.dist.parallel_state import is_enable_sequence_parallel
        if not is_enable_sequence_parallel():
            while True:
                try:
                    return self.getitem(index)
                except Exception as e:
                    print("--------------------------------- read data error: ", e)
                    index, duration, height, width = [int(val) for val in index.split("-")]
                    #index, duration, height, width
                    if duration == 0: #图片
                        indices = self.data[self.data['duration'] == duration].index
                    else:
                        indices = self.data[self.data['duration'] >= duration].index
                        if len(indices) == 0: #避免没有对应的时长视频，可能行较小
                            indices = self.data[self.data['duration'] != 0].index
                    random_index = np.random.choice(indices) #这个random可能会影响不同卡上操作的随机性，但是这本来就应该如此，只要单个rank上操作是一致的就行，只要写好seed就行吧
                    index = f"{random_index}-{duration}-{height}-{width}"
        else:
            valid = torch.ones(1).bool()
            data = {}

            try:                
                data = self.getitem(index)
            except Exception as e:
                valid = torch.zeros(1).bool()
                
                from training_acc.logger import logger
                from training_acc.dist import log_rank
                logger.info(log_rank(f"--------------------------------- read data index:{index}, error: {e}"))
                
            data["valid"] = valid
            return data


@DATASETS.register_module()
class VariableVideoFlowTextWithDurationDataset(VariableVideoTextWithDurationDataset):
    """
    扩展版数据集：在 VariableVideoTextWithDurationDataset 的基础上，
    同时读取 video_root 和 flow_root 下的两段视频（同名文件），
    做相同的时域采样 + 相同的空间变换，返回:
      - ret["video"]: [C, T, H, W]
      - ret["flow"]:  [C, T, H, W]
    其它键（text、height、width 等）保持不变。
    """

    def __init__(
        self,
        video_root: str,
        flow_root: str,
        **kwargs,
    ):
        # 复用父类的初始化逻辑
        super().__init__(**kwargs)
        self.video_root = video_root
        self.flow_root = flow_root

    def getitem(self, index):
        # 和父类一样，先解析 index 里的 "i-duration-H-W"
        index, duration, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]

        # -------- 1) 拿到原始 path / text（完全复用父类逻辑） --------
        if "path" not in sample:
            # lmdb 分支
            key = sample["key"]
            with self.env.begin(write=False) as txn:
                info = pickle.loads(txn.get(key.encode("ascii")))
            raw_path, text = info
        else:
            raw_path = sample["path"]
            # 原代码里有一行替换，这里保持不动（防止你其它数据也在用）
            raw_path = raw_path.replace(
                "/mnt/sh_nas/moyuan.yty/", "/home/dufei.df/huaniu_workspace/"
            )

        # 用 basename 拼接 video / flow 的实际路径
        basename = os.path.basename(raw_path)
        video_path = os.path.join(self.video_root, basename)
        flow_path = os.path.join(self.flow_root, basename)

        file_type = self.get_type(video_path)
        ar = height / width

        if file_type == "video":
            # ========== 2) 打开 video / flow 两个 Reader ==========
            video_fp = read(video_path)
            video_reader = decord.VideoReader(video_fp)

            flow_fp = read(flow_path)
            flow_reader = decord.VideoReader(flow_fp)

            ori_fps = video_reader.get_avg_fps()
            ori_video_length = len(video_reader)

            # 下面这块完全照父类逻辑抄：算 num_frames / 索引
            num_frames = int(duration * self.sample_fps)
            if self.add_one:
                num_frames = num_frames + 1
            required_len = math.ceil(num_frames / self.sample_fps * ori_fps)

            clip_len = min(10, max(ori_video_length - required_len, 0) // 2)

            normed_video_length = round(
                (ori_video_length - 2 * clip_len) / ori_fps * self.sample_fps
            )
            normed_video_length = max(num_frames, normed_video_length)

            batch_index_all = (
                np.linspace(
                    clip_len, ori_video_length - 1 - clip_len, normed_video_length
                )
                .round()
                .astype(int)
            )
            start_idx = 0  # 如果你想随机，可以改回 random.randint
            batch_index = batch_index_all[start_idx : start_idx + num_frames]

            # T, H, W, C -> T, C, H, W
            video = video_reader.get_batch(batch_index).permute(0, 3, 1, 2)
            flow = flow_reader.get_batch(batch_index).permute(0, 3, 1, 2)

            video_fps = self.sample_fps

            # ========== 3) 用同一个 transform 对 video & flow 做相同空间变换 ==========
            transform = get_transforms_video(self.transform_name, (height, width))
            # 拼接在时间维：[2T, C, H, W]，确保一次 transform 的随机 crop 是一致的
            vf = torch.cat([video, flow], dim=0)
            vf = transform(vf)  # [2T, C, H', W']

            video = vf[:num_frames]   # [T, C, H', W']
            flow = vf[num_frames:]    # [T, C, H', W']

        else:
            # 理论上你 mixkit 都是视频，这里做个兜底
            if raw_path.startswith("oss"):
                image = read_pil_image(raw_path)
            else:
                image = pil_loader(raw_path)
            video_fps = IMG_FPS

            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)
            num_frames = 1
            video = image.unsqueeze(0)
            flow = torch.zeros_like(video)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        flow = flow.permute(1, 0, 2, 3)

        ret = {
            "video": video,
            "flow": flow,               # ★ 新增：光流
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
        }

        if self.get_text:
            ret["text"] = sample["text"]
        else:
            ret["text"] = text

        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len

        return ret


@DATASETS.register_module()
class BatchFeatureDataset(torch.utils.data.Dataset):
    """
    The dataset is composed of multiple .bin files.
    Each .bin file is a list of batch data (like a buffer). All .bin files have the same length.
    In each training iteration, one batch is fetched from the current buffer.
    Once a buffer is consumed, load another one.
    Avoid loading the same .bin on two difference GPUs, i.e., one .bin is assigned to one GPU only.
    """

    def __init__(self, data_path=None):
        self.path_list = sorted(glob(data_path + "/**/*.bin"))

        self._len_buffer = len(torch.load(self.path_list[0]))
        self._num_buffers = len(self.path_list)
        self.num_samples = self.len_buffer * len(self.path_list)

        self.cur_file_idx = -1
        self.cur_buffer = None

    @property
    def num_buffers(self):
        return self._num_buffers

    @property
    def len_buffer(self):
        return self._len_buffer

    def _load_buffer(self, idx):
        file_idx = idx // self.len_buffer
        if file_idx != self.cur_file_idx:
            self.cur_file_idx = file_idx
            self.cur_buffer = torch.load(self.path_list[file_idx])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._load_buffer(idx)

        batch = self.cur_buffer[idx % self.len_buffer]  # dict; keys are {'x', 'fps'} and text related

        ret = {
            "video": batch["x"],
            "text": batch["y"],
            "mask": batch["mask"],
            "fps": batch["fps"],
            "height": batch["height"],
            "width": batch["width"],
            "num_frames": batch["num_frames"],
        }
        return ret

@DATASETS.register_module()
class VariableVideoTextPerRankDataset(torch.utils.data.Dataset):
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
        
        bucket_id, path, text = sample['bucket_id'], sample['path'], sample['text']
        
        duration, height, width, _ = [int(val) for val in bucket_id.split("-")]
        
        path = path.replace("/mnt/sh_nas/moyuan.yty/", "/home/dufei.df/huaniu_workspace/")
        # path = path.replace("oss://", "/root/")
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
        ret = {
            "video": video,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
            "text": text
        }
        
        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len
        return ret

    def __getitem__(self, index): 
        from training_acc.dist.parallel_state import is_enable_sequence_parallel
        if not is_enable_sequence_parallel():
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
        else:
            valid = torch.ones(1).bool()
            data = {}

            try:                
                data = self.getitem(index)
            except Exception as e:
                valid = torch.zeros(1).bool()
                
                from training_acc.logger import logger
                from training_acc.dist import log_rank
                logger.info(log_rank(f"--------------------------------- read data index:{index}, error: {e}"))
                
            data["valid"] = valid
            return data
            
