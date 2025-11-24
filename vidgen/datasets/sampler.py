from collections import OrderedDict, defaultdict
from pprint import pformat
from typing import Iterator, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler

from vidgen.utils.misc import format_numel_str, get_logger
from copy import deepcopy
from .aspect import get_num_pixels
from .bucket import Bucket, SecondsBucket
from .datasets import VariableVideoTextDataset, VariableVideoTextWithDurationDataset, VariableVideoTextPerRankDataset


# use pandarallel to accelerate bucket processing
# NOTE: pandarallel should only access local variables
def apply(data, method=None, frame_interval=None, seed=None, num_bucket=None):
    return method(
        data["num_frames"],
        data["height"],
        data["width"],
        frame_interval,
        seed + data["id"] * num_bucket,
    )
    
def apply_by_duration(data, method=None, seed=None, num_bucket=None):
    return method(
        data["duration"],
        data["height"],
        data["width"],
        seed + data["id"] * num_bucket,
    )

class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def reset(self) -> None:
        self.start_index = 0

    def state_dict(self, step) -> dict:
        return {"start_index": step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)


class VariableVideoBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset: VariableVideoTextDataset,
        bucket_config: dict,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
        num_bucket_build_workers: int = 1,
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
        self.dataset = dataset
        self.bucket = Bucket(bucket_config)
        self.verbose = verbose
        self.last_micro_batch_access_index = 0
        self.approximate_num_batch = None

        self._get_num_batch_cached_bucket_sample_dict = None
        self.num_bucket_build_workers = num_bucket_build_workers

    def __iter__(self) -> Iterator[List[int]]:
        if self._get_num_batch_cached_bucket_sample_dict is not None:
            bucket_sample_dict = self._get_num_batch_cached_bucket_sample_dict
            self._get_num_batch_cached_bucket_sample_dict = None
        else:
            bucket_sample_dict = self.group_by_bucket()
            if self.verbose:
                self._print_bucket_info(bucket_sample_dict)

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_micro_batch_count = OrderedDict()
        bucket_last_consumed = OrderedDict()

        # process the samples
        for bucket_id, data_list in bucket_sample_dict.items():
            # handle droplast
            bs_per_gpu = self.bucket.get_batch_size(bucket_id)
            total_bs = bs_per_gpu * self.num_replicas
            remainder = len(data_list) % total_bs #保证每个bucket id里的data_list都能被bs_per_gpu * self.num_replicas整除

            if remainder > 0:
                if not self.drop_last:
                    # if there is remainder, we pad to make it divisible
                    data_list += data_list[: total_bs - remainder]
                else:
                    # we just drop the remainder to make it divisible
                    data_list = data_list[:-remainder]
            
            bucket_sample_dict[bucket_id] = data_list 

            # handle shuffle
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in data_indices]
                bucket_sample_dict[bucket_id] = data_list

            # compute how many micro-batches each bucket has
            num_micro_batches = len(data_list) // total_bs
            bucket_micro_batch_count[bucket_id] = num_micro_batches

        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in bucket_micro_batch_count.items(): #这个batch是所有机器上的大batch数目
            bucket_id_access_order.extend([bucket_id] * num_micro_batch)

        # randomize the access order
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(len(bucket_id_access_order), generator=g).tolist()
            bucket_id_access_order = [bucket_id_access_order[i] for i in bucket_id_access_order_indices]

        # prepare each batch from its bucket
        # according to the predefined bucket access order
        num_iters = len(bucket_id_access_order)
        start_iter_idx = self.last_micro_batch_access_index

        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of GPUs
        
        for i in range(self.last_micro_batch_access_index):
            bucket_id = bucket_id_access_order[i]
            bucket_bs = self.bucket.get_batch_size(bucket_id)
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs * self.num_replicas
            else:
                bucket_last_consumed[bucket_id] = bucket_bs * self.num_replicas

        for i in range(start_iter_idx, num_iters):
            bucket_id = bucket_id_access_order[i]
            self.last_micro_batch_access_index += 1

            # compute the data samples consumed by each access
            
            bucket_bs = self.bucket.get_batch_size(bucket_id)
            last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
            
            bucket_access_boundaries = []
            for j in range(self.num_replicas):
                bucket_access_boundaries.append([last_consumed_index + j * bucket_bs, last_consumed_index + bucket_bs * (j+1)])

                # update consumption
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs * self.num_replicas
            else:
                bucket_last_consumed[bucket_id] = bucket_bs * self.num_replicas

            # compute the range of data accessed by each GPU
            boundary = bucket_access_boundaries[self.rank]
            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0] : boundary[1]]

            # encode t, h, w into the sample index
            real_t, real_h, real_w = self.bucket.get_thw(bucket_id)
            cur_micro_batch = [f"{idx}-{real_t}-{real_h}-{real_w}" for idx in cur_micro_batch]
            yield cur_micro_batch

        self.reset()

    def __len__(self) -> int:
        return self.get_num_batch() // self.num_replicas

    def group_by_bucket(self) -> dict:
        bucket_sample_dict = OrderedDict()

        from pandarallel import pandarallel

        pandarallel.initialize(nb_workers=self.num_bucket_build_workers, progress_bar=False)
        get_logger().info("Building buckets...")
        bucket_ids = self.dataset.data.parallel_apply(
            apply,
            axis=1,
            method=self.bucket.get_bucket_id,
            frame_interval=self.dataset.frame_interval,
            seed=self.seed + self.epoch,
            num_bucket=self.bucket.num_bucket,
        )

        # group by bucket
        # each data sample is put into a bucket with a similar image/video size
        for i in range(len(self.dataset)):
            bucket_id = bucket_ids[i] #hw_id, t_id, ar_id
            if bucket_id is None:
                continue
            if bucket_id not in bucket_sample_dict:
                bucket_sample_dict[bucket_id] = []
            bucket_sample_dict[bucket_id].append(i)
        return bucket_sample_dict

    def get_num_batch(self) -> int:
        bucket_sample_dict = self.group_by_bucket() #以hw_id, t_id, ar_id为key的dict
        self._get_num_batch_cached_bucket_sample_dict = bucket_sample_dict

        # calculate the number of batches
        if self.verbose:
            self._print_bucket_info(bucket_sample_dict)
        return self.approximate_num_batch

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        # collect statistics
        total_samples = 0
        total_batch = 0
        num_aspect_dict = defaultdict(lambda: [0, 0])
        num_hwt_dict = defaultdict(lambda: [0, 0])
        
        for k, v in bucket_sample_dict.items():
            size = len(v)
            remainder = size % (self.bucket.get_batch_size(k[:-1]) * self.num_replicas)
            
            if remainder > 0:
                if self.drop_last:
                    size = size - remainder
                else:
                    size = size + (self.bucket.get_batch_size(k[:-1]) * self.num_replicas - remainder)

            num_batch = size // self.bucket.get_batch_size(k[:-1])

            total_samples += size
            total_batch += num_batch

            num_aspect_dict[k[-1]][0] += size
            num_aspect_dict[k[-1]][1] += num_batch
            num_hwt_dict[k[:-1]][0] += size
            num_hwt_dict[k[:-1]][1] += num_batch

        
        # sort
        num_aspect_dict = dict(sorted(num_aspect_dict.items(), key=lambda x: x[0]))
        num_hwt_dict = dict(
            sorted(num_hwt_dict.items(), key=lambda x: (get_num_pixels(x[0][0]), x[0][1]), reverse=True)
        )
        num_hwt_img_dict = {k: v for k, v in num_hwt_dict.items() if k[1] == 1}
        num_hwt_vid_dict = {k: v for k, v in num_hwt_dict.items() if k[1] > 1}

        # log
        if dist.get_rank() == 0 and self.verbose:
            get_logger().info("Bucket Info:")
            get_logger().info(
                "Bucket [#sample, #batch] by aspect ratio:\n%s", pformat(num_aspect_dict, sort_dicts=False)
            )
            get_logger().info(
                "Image Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_img_dict, sort_dicts=False)
            )
            get_logger().info(
                "Video Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_vid_dict, sort_dicts=False)
            )
            get_logger().info(
                "#training batch: %s, #training sample: %s, #non empty bucket: %s",
                format_numel_str(total_batch),
                format_numel_str(total_samples),
                len(bucket_sample_dict),
            )
        self.approximate_num_batch = total_batch

    def reset(self):
        self.last_micro_batch_access_index = 0

    def state_dict(self, num_steps: int) -> dict:
        # the last_micro_batch_access_index in the __iter__ is often
        # not accurate during multi-workers and data prefetching
        # thus, we need the user to pass the actual steps which have been executed
        # to calculate the correct last_micro_batch_access_index
        return {"seed": self.seed, "epoch": self.epoch, "last_micro_batch_access_index": num_steps}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)


class VariableVideoBatchBalanceWithDurationSampler(DistributedSampler):
    def __init__(
        self,
        dataset: VariableVideoTextWithDurationDataset,
        bucket_config: dict,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
        num_bucket_build_workers: int = 1,
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
        self.dataset = dataset
        self.bucket = SecondsBucket(bucket_config)
        self.verbose = verbose
        self.last_micro_batch_access_index = 0
        self.approximate_num_batch = None

        self._get_num_batch_cached_bucket_sample_dict = None
        self.num_bucket_build_workers = num_bucket_build_workers

    def __iter__(self) -> Iterator[List[int]]:
        if self._get_num_batch_cached_bucket_sample_dict is not None:
            bucket_sample_dict = self._get_num_batch_cached_bucket_sample_dict
            self._get_num_batch_cached_bucket_sample_dict = None
        else:
            bucket_sample_dict = self.group_by_bucket()
            if self.verbose:
                self._print_bucket_info(bucket_sample_dict)

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_micro_batch_count = OrderedDict()
        bucket_last_consumed = OrderedDict()

        # process the samples
        for bucket_id, data_list in bucket_sample_dict.items():
            # handle droplast
            bs_per_gpu = self.bucket.get_batch_size(bucket_id)
            total_bs = bs_per_gpu * self.num_replicas
            remainder = len(data_list) % total_bs #保证每个bucket id里的data_list都能被bs_per_gpu * self.num_replicas整除

            if remainder > 0:
                if not self.drop_last:
                    # if there is remainder, we pad to make it divisible
                    data_list += data_list[: total_bs - remainder]
                else:
                    # we just drop the remainder to make it divisible
                    data_list = data_list[:-remainder]
            
            bucket_sample_dict[bucket_id] = data_list 

            # handle shuffle
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in data_indices]
                bucket_sample_dict[bucket_id] = data_list

            # compute how many micro-batches each bucket has
            num_micro_batches = len(data_list) // total_bs
            bucket_micro_batch_count[bucket_id] = num_micro_batches

        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in bucket_micro_batch_count.items(): #这个batch是所有机器上的大batch数目
            bucket_id_access_order.extend([bucket_id] * num_micro_batch)

        # randomize the access order
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(len(bucket_id_access_order), generator=g).tolist()
            bucket_id_access_order = [bucket_id_access_order[i] for i in bucket_id_access_order_indices]

        # prepare each batch from its bucket
        # according to the predefined bucket access order
        num_iters = len(bucket_id_access_order)
        start_iter_idx = self.last_micro_batch_access_index

        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of GPUs
        
        for i in range(self.last_micro_batch_access_index):
            bucket_id = bucket_id_access_order[i]
            bucket_bs = self.bucket.get_batch_size(bucket_id)
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs * self.num_replicas
            else:
                bucket_last_consumed[bucket_id] = bucket_bs * self.num_replicas

        for i in range(start_iter_idx, num_iters):
            bucket_id = bucket_id_access_order[i]
            self.last_micro_batch_access_index += 1

            # compute the data samples consumed by each access
            
            bucket_bs = self.bucket.get_batch_size(bucket_id)
            last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
            
            bucket_access_boundaries = []
            for j in range(self.num_replicas):
                bucket_access_boundaries.append([last_consumed_index + j * bucket_bs, last_consumed_index + bucket_bs * (j+1)])

                # update consumption
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs * self.num_replicas
            else:
                bucket_last_consumed[bucket_id] = bucket_bs * self.num_replicas

            # compute the range of data accessed by each GPU
            boundary = bucket_access_boundaries[self.rank]
            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0] : boundary[1]]

            # encode t, h, w into the sample index
            real_t, real_h, real_w = self.bucket.get_thw(bucket_id)
            cur_micro_batch = [f"{idx}-{real_t}-{real_h}-{real_w}" for idx in cur_micro_batch]
            yield cur_micro_batch

        self.reset()

    def __len__(self) -> int:
        return self.get_num_batch() // self.num_replicas

    def group_by_bucket(self) -> dict:
        bucket_sample_dict = OrderedDict()

        from pandarallel import pandarallel

        pandarallel.initialize(nb_workers=self.num_bucket_build_workers, progress_bar=False)
        get_logger().info("Building buckets...")
        bucket_ids = self.dataset.data.parallel_apply(
            apply_by_duration,
            axis=1,
            method=self.bucket.get_bucket_id,
            seed=self.seed + self.epoch,
            num_bucket=self.bucket.num_bucket,
        )

        # group by bucket
        # each data sample is put into a bucket with a similar image/video size
        for i in range(len(self.dataset)):
            bucket_id = bucket_ids[i] #hw_id, t_id, ar_id
            if bucket_id is None:
                continue
            if bucket_id not in bucket_sample_dict:
                bucket_sample_dict[bucket_id] = []
            bucket_sample_dict[bucket_id].append(i)
        return bucket_sample_dict

    def get_num_batch(self) -> int:
        bucket_sample_dict = self.group_by_bucket() #以hw_id, t_id, ar_id为key的dict
        self._get_num_batch_cached_bucket_sample_dict = bucket_sample_dict

        # calculate the number of batches
        if self.verbose:
            self._print_bucket_info(bucket_sample_dict)
        return self.approximate_num_batch

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        # collect statistics
        total_samples = 0
        total_batch = 0
        num_aspect_dict = defaultdict(lambda: [0, 0])
        num_hwt_dict = defaultdict(lambda: [0, 0])
        
        for k, v in bucket_sample_dict.items():
            size = len(v)
            remainder = size % (self.bucket.get_batch_size(k[:-1]) * self.num_replicas)
            
            if remainder > 0:
                if self.drop_last:
                    size = size - remainder
                else:
                    size = size + (self.bucket.get_batch_size(k[:-1]) * self.num_replicas - remainder)

            num_batch = size // self.bucket.get_batch_size(k[:-1])

            total_samples += size
            total_batch += num_batch

            num_aspect_dict[k[-1]][0] += size
            num_aspect_dict[k[-1]][1] += num_batch
            num_hwt_dict[k[:-1]][0] += size
            num_hwt_dict[k[:-1]][1] += num_batch

        
        # sort
        num_aspect_dict = dict(sorted(num_aspect_dict.items(), key=lambda x: x[0]))
        num_hwt_dict = dict(
            sorted(num_hwt_dict.items(), key=lambda x: (get_num_pixels(x[0][0]), x[0][1]), reverse=True)
        )
        num_hwt_img_dict = {k: v for k, v in num_hwt_dict.items() if k[1] == 0}
        num_hwt_vid_dict = {k: v for k, v in num_hwt_dict.items() if k[1] > 0}

        # log
        if dist.get_rank() == 0 and self.verbose:
            get_logger().info("Bucket Info:")
            get_logger().info(
                "Bucket [#sample, #batch] by aspect ratio:\n%s", pformat(num_aspect_dict, sort_dicts=False)
            )
            get_logger().info(
                "Image Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_img_dict, sort_dicts=False)
            )
            get_logger().info(
                "Video Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_vid_dict, sort_dicts=False)
            )
            get_logger().info(
                "#training batch: %s, #training sample: %s, #non empty bucket: %s",
                format_numel_str(total_batch),
                format_numel_str(total_samples),
                len(bucket_sample_dict),
            )
        self.approximate_num_batch = total_batch

    def reset(self):
        self.last_micro_batch_access_index = 0

    def state_dict(self, num_steps: int) -> dict:
        # the last_micro_batch_access_index in the __iter__ is often
        # not accurate during multi-workers and data prefetching
        # thus, we need the user to pass the actual steps which have been executed
        # to calculate the correct last_micro_batch_access_index
        return {"seed": self.seed, "epoch": self.epoch, "last_micro_batch_access_index": num_steps}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)
        
class BatchDistributedSampler(DistributedSampler):
    """
    Used with BatchDataset;
    Suppose len_buffer == 5, num_buffers == 6, #GPUs == 3, then
           | buffer {i}          | buffer {i+1}
    ------ | ------------------- | -------------------
    rank 0 |  0,  1,  2,  3,  4, |  5,  6,  7,  8,  9
    rank 1 | 10, 11, 12, 13, 14, | 15, 16, 17, 18, 19
    rank 2 | 20, 21, 22, 23, 24, | 25, 26, 27, 28, 29
    """

    def __init__(self, dataset: Dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.start_index = 0

    def __iter__(self):
        num_buffers = self.dataset.num_buffers
        len_buffer = self.dataset.len_buffer
        num_buffers_i = num_buffers // self.num_replicas
        num_samples_i = len_buffer * num_buffers_i

        indices_i = np.arange(self.start_index, num_samples_i) + self.rank * num_samples_i
        indices_i = indices_i.tolist()

        return iter(indices_i)

    def reset(self):
        self.start_index = 0

    def state_dict(self, step) -> dict:
        return {"start_index": step}

    def load_state_dict(self, state_dict: dict):
        self.start_index = state_dict["start_index"] + 1

class VariableVideoBatchPerRankSampler(DistributedSampler):
    def __init__(
        self,
        dataset: VariableVideoTextPerRankDataset,
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

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)