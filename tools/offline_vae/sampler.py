from collections import OrderedDict, defaultdict
from pprint import pformat
from typing import Iterator, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler

from vidgen.utils.misc import format_numel_str, get_logger
from copy import deepcopy

class VariableVideoBatchPerRankSamplerVae(DistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
        self.dataset = dataset
        self.verbose = verbose
        self.last_micro_batch_access_index = 0
        self.approximate_num_batch = 0
        # self.bucket_id_list = sorted(dataset.data['bucket_id'].drop_duplicates().to_list())
        # self.bucket_id_num_count = dataset.data['bucket_id'].value_counts()
        image_percent = dataset.image_percent
        self.bucket_id_video_list = [bucket_id for bucket_id in dataset.bucket_id_list if int(bucket_id.split("-")[0]) != 0]
        self.bucket_id_image_list = [bucket_id for bucket_id in dataset.bucket_id_list if int(bucket_id.split("-")[0]) == 0]
        self.bucket_id_video_num_count = {key: dataset.bucket_id_num_count[key] for key in self.bucket_id_video_list}
        self.bucket_id_image_num_count = {key: dataset.bucket_id_num_count[key] for key in self.bucket_id_image_list}
        
        approximate_video_num_batch = 0
        approximate_image_num_batch = 0
        for bucket_id in self.bucket_id_video_list:
            bucket_bs = int(bucket_id.split("-")[-1])
            approximate_video_num_batch += self.bucket_id_video_num_count[bucket_id] // bucket_bs
        
        for bucket_id in self.bucket_id_image_list:
            bucket_bs = int(bucket_id.split("-")[-1])
            approximate_image_num_batch += self.bucket_id_image_num_count[bucket_id] // bucket_bs
        
        self.remain_image_ratio = 1
        
        if image_percent is not None:
            expected_image_num_batch = int(approximate_video_num_batch * image_percent) if approximate_video_num_batch != 0 else approximate_image_num_batch
            if approximate_image_num_batch > expected_image_num_batch:
                self.remain_image_ratio = expected_image_num_batch / approximate_image_num_batch
                approximate_image_num_batch = expected_image_num_batch
        
        self.approximate_image_num_batch = approximate_image_num_batch
        self.approximate_num_batch = approximate_video_num_batch + approximate_image_num_batch
        
        if self.verbose:
            num_hwt_img_dict = {}
            num_hwt_vid_dict = {}
            for bucket_id in self.bucket_id_video_list:
                bucket_bs = int(bucket_id.split("-")[-1])
                num_hwt_vid_dict[bucket_id] = [self.bucket_id_video_num_count[bucket_id], self.bucket_id_video_num_count[bucket_id] // bucket_bs]

            for bucket_id in self.bucket_id_image_list:
                bucket_bs = int(bucket_id.split("-")[-1])
                num_hwt_img_dict[bucket_id] = [self.bucket_id_image_num_count[bucket_id], self.bucket_id_image_num_count[bucket_id] // bucket_bs]
        
            get_logger().info("Bucket Info:")
            get_logger().info(
                "Image Bucket [#sample, #batch]:\n%s", pformat(num_hwt_img_dict, sort_dicts=False)
            )
            get_logger().info(
                "Video Bucket [#sample, #batch]:\n%s", pformat(num_hwt_vid_dict, sort_dicts=False)
            )
            get_logger().info(
                "Image iter is %s of video iter, Sample image ratio : %s\n", image_percent, self.remain_image_ratio
            )
                        
    def __iter__(self) -> Iterator[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_micro_batch_count = {}
        bucket_last_consumed = {}
        bucket_sample_dict = {}
        for bucket_id in self.bucket_id_video_list:
            data_list = torch.tensor(self.dataset.data.index[self.dataset.data['bucket_id'] == bucket_id])
            bucket_sample_dict[bucket_id] = data_list
        
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g)
                bucket_sample_dict[bucket_id] = bucket_sample_dict[bucket_id][data_indices]

            bucket_bs = int(bucket_id.split("-")[-1])
            # compute how many micro-batches each bucket has
            num_micro_batches = len(data_list) // bucket_bs
            bucket_micro_batch_count[bucket_id] = num_micro_batches

        for bucket_id in self.bucket_id_image_list:
            data_list = torch.tensor(self.dataset.data.index[self.dataset.data['bucket_id'] == bucket_id])
            bucket_bs = int(bucket_id.split("-")[-1])
            last_index = int((len(data_list) * self.remain_image_ratio) // bucket_bs * bucket_bs)
            if last_index == 0:
                continue
            bucket_sample_dict[bucket_id] = data_list[:last_index]
        
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g)
                data_list = data_list[data_indices]
                bucket_sample_dict[bucket_id] = data_list[:last_index]
            
            # compute how many micro-batches each bucket has
            num_micro_batches = len(bucket_sample_dict[bucket_id]) // bucket_bs
            bucket_micro_batch_count[bucket_id] = num_micro_batches
        
        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id2str_dict = {}
        bucket_str2id_dict = {}
        for i, key in enumerate(bucket_micro_batch_count.keys()):
            bucket_id2str_dict[i] = key
            bucket_str2id_dict[key] = i
        
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in bucket_micro_batch_count.items(): #这个batch是所有机器上的大batch数目
            num_id = bucket_str2id_dict[bucket_id]
            bucket_id_access_order.extend([num_id] * num_micro_batch)
        
        bucket_id_access_order = torch.tensor(bucket_id_access_order)
        
        # randomize the access order
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(len(bucket_id_access_order), generator=g)
            bucket_id_access_order = bucket_id_access_order[bucket_id_access_order_indices]
        
        # prepare each batch from its bucket
        # according to the predefined bucket access order
        num_iters = len(bucket_id_access_order)
        start_iter_idx = self.last_micro_batch_access_index

        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of GPUs
        
        for i in range(self.last_micro_batch_access_index):
            bucket_id = bucket_id2str_dict[bucket_id_access_order[i].item()]
            bucket_bs = int(bucket_id.split("-")[-1])
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs
            else:
                bucket_last_consumed[bucket_id] = bucket_bs

        for i in range(start_iter_idx, num_iters):
            bucket_id = bucket_id2str_dict[bucket_id_access_order[i].item()]
            self.last_micro_batch_access_index += 1

            # compute the data samples consumed by each access
            
            bucket_bs = int(bucket_id.split("-")[-1])
            last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
            
            boundary = [last_consumed_index, last_consumed_index + bucket_bs]
                # update consumption
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs
            else:
                bucket_last_consumed[bucket_id] = bucket_bs

            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0] : boundary[1]].tolist()

            # encode t, h, w into the sample index
            # cur_micro_batch = [f"{bucket_id}_{idx}" for idx in cur_micro_batch]
            yield cur_micro_batch
        
        self.reset()

    def __len__(self) -> int:
        return self.approximate_num_batch

    def reset(self):
        self.last_micro_batch_access_index = 0

    def state_dict(self, num_steps: int) -> dict:
        return {"seed": self.seed, "epoch": self.epoch, "last_micro_batch_access_index": num_steps}

    def set_start_index(self, index):
        self.last_micro_batch_access_index = index
        
    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)