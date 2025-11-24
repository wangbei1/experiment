
from collections import OrderedDict, defaultdict
from pprint import pformat
from typing import Iterator, List, Optional

from vidgen.datasets.aspect import ASPECT_RATIOS, get_closest_ratio
from functools import partial
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.distributed as dist
import json
from vidgen.datasets.utils import read_file
from vidgen.datasets.aspect import get_num_pixels
import pandas as pd
import time
from multiprocessing import Pool

def apply_by_duration(data, method=None, seed=None, num_bucket=None):
    #"id", 'duration', 'height', 'width'顺序
    return method(
        data[1],
        data[2],
        data[3],
        seed + data[0] * num_bucket,
    )

class SecondsBucket:
    def __init__(self, bucket_config):
        for key in bucket_config:
            assert key in ASPECT_RATIOS, f"Aspect ratio {key} not found."
        # wrap config with OrderedDict
        bucket_probs = OrderedDict()
        bucket_bs = OrderedDict()
        bucket_names = sorted(bucket_config.keys(), key=lambda x: ASPECT_RATIOS[x][0], reverse=True)
        for key in bucket_names:
            bucket_time_names = sorted(bucket_config[key].keys(), key=lambda x: x, reverse=True)
            bucket_probs[key] = OrderedDict({k: bucket_config[key][k][0] for k in bucket_time_names})
            bucket_bs[key] = OrderedDict({k: bucket_config[key][k][1] for k in bucket_time_names})

        # first level: HW
        num_bucket = 0
        hw_criteria = dict()
        t_criteria = dict()
        ar_criteria = dict()
        bucket_id = OrderedDict()
        bucket_id_cnt = 0
        for k1, v1 in bucket_probs.items():
            hw_criteria[k1] = ASPECT_RATIOS[k1][0]
            t_criteria[k1] = dict()
            ar_criteria[k1] = dict()
            bucket_id[k1] = dict()
            for k2, _ in v1.items():
                t_criteria[k1][k2] = k2
                bucket_id[k1][k2] = bucket_id_cnt
                bucket_id_cnt += 1
                ar_criteria[k1][k2] = dict()
                for k3, v3 in ASPECT_RATIOS[k1][1].items():
                    ar_criteria[k1][k2][k3] = v3
                    num_bucket += 1

        self.bucket_probs = bucket_probs
        self.bucket_bs = bucket_bs
        self.bucket_id = bucket_id
        self.hw_criteria = hw_criteria
        self.t_criteria = t_criteria
        self.ar_criteria = ar_criteria
        self.num_bucket = num_bucket
        print(f"Number of buckets: {num_bucket}")

    def get_bucket_id(self, T, H, W, seed=None):
        resolution = H * W

        fail = True
        for hw_id, t_criteria in self.bucket_probs.items():
            if resolution < self.hw_criteria[hw_id]:
                continue

            # if sample is an image
            if T == 0: #这里0作为图片的s数
                if 0 in t_criteria:
                    fail = False
                    t_id = 0
                    break
                else:
                    continue

            # otherwise, find suitable t_id for video
            t_fail = True
            for t_id, prob in t_criteria.items():
                if T >= t_id and t_id != 0: #可以等于
                    t_fail = False
                    break
            if t_fail:
                continue

            fail = False
            break
        
        if fail:
            return None

        # get aspect ratio id
        ar_criteria = self.ar_criteria[hw_id][t_id]
        ar_id = get_closest_ratio(H, W, ar_criteria)
        return hw_id, t_id, ar_id

    def get_thw(self, bucket_id):
        assert len(bucket_id) == 3
        T = self.t_criteria[bucket_id[0]][bucket_id[1]]
        H, W = self.ar_criteria[bucket_id[0]][bucket_id[1]][bucket_id[2]]
        return T, H, W

    def get_prob(self, bucket_id):
        return self.bucket_probs[bucket_id[0]][bucket_id[1]]

    def get_batch_size(self, bucket_id):
        return self.bucket_bs[bucket_id[0]][bucket_id[1]]

    def __len__(self):
        return self.num_bucket
    
class Sampler():
    def __init__(
        self,
        data,
        bucket_config: dict,
        num_replicas,
        shuffle: bool = True,
        seed: int = 1024,
        drop_last: bool = False,
        verbose: bool = False,
        num_bucket_build_workers: int = 16
    ):
        
        self.data = data
        self.bucket = SecondsBucket(bucket_config)
        self.verbose = verbose
        
        self.approximate_num_batch = None
        self.seed = seed
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.drop_last = drop_last
        self.num_bucket_build_workers = num_bucket_build_workers
    
    def group_by_bucket(self) -> dict:
        bucket_sample_dict = OrderedDict()
        
        start_time = time.time()
        print("Building buckets...")
        
        sub_data = self.data.loc[:, ["id", 'duration', 'height', 'width']].values.tolist()
        
        # sub_data = [sub_data.iloc[i] for i in range(len(sub_data))]
        
        with Pool(processes=48) as pool:
            results = pool.map(
                partial(
                    apply_by_duration,
                    method=self.bucket.get_bucket_id,
                    seed=self.seed,
                    num_bucket=self.bucket.num_bucket,
                ),
                sub_data,
            )
        
        # bucket_ids = self.data.loc[:, ["id", 'duration', 'height', 'width']].parallel_apply(
        #     apply_by_duration,
        #     axis=1,
        #     method=self.bucket.get_bucket_id,
        #     seed=self.seed,
        #     num_bucket=self.bucket.num_bucket,
        # )
        bucket_ids = results

        print("finush building buckets. use time: ", time.time() - start_time)
        # import pdb
        # pdb.set_trace()
        # group by bucket
        # each data sample is put into a bucket with a similar image/video size
        for i in range(len(self.data)):
            bucket_id = bucket_ids[i] #hw_id, t_id, ar_id
            if bucket_id is None:
                continue
            if bucket_id not in bucket_sample_dict:
                bucket_sample_dict[bucket_id] = []
            bucket_sample_dict[bucket_id].append(i)
        return bucket_sample_dict
    
    def get_split(self):
        bucket_sample_dict = self.group_by_bucket()
        if self.verbose:
            self._print_bucket_info(bucket_sample_dict)

        g = torch.Generator()
        g.manual_seed(self.seed)
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

        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of GPUs
        
        #保证同一个bucket所有卡都是分布均匀的，这样采样的时候能够有效地采集
        #然后可以sampler时根据bucket_id采样，bs是确定的，其实就是采哪个bucket，哪个数据，bucket内部可以shuffle
        #采样的时候就是采哪个bucket，哪个index，这样就能对得上了
        bucket_id_to_index = OrderedDict()
        for i in range(self.num_replicas):
            bucket_id_to_index[i] = defaultdict(list)
        
        for i in tqdm(range(num_iters)):
            bucket_id = bucket_id_access_order[i]

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
            
            for j in range(len(bucket_access_boundaries)):
                boundary = bucket_access_boundaries[j]
                cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0] : boundary[1]]
                real_t, real_h, real_w = self.bucket.get_thw(bucket_id)
                
                for idx in cur_micro_batch:
                    bucket_id_to_index[j][f"{real_t}-{real_h}-{real_w}-{bucket_bs}"].append(idx)

        return bucket_id_to_index
    
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
        if self.verbose:
            print("Bucket Info:")
            print(
                "Bucket [#sample, #batch] by aspect ratio:\n%s", pformat(num_aspect_dict, sort_dicts=False)
            )
            print(
                "Image Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_img_dict, sort_dicts=False)
            )
            print(
                "Video Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_hwt_vid_dict, sort_dicts=False)
            )
            print(
                f"#training batch: {total_batch}, #training sample: {total_samples}, #non empty bucket: {len(bucket_sample_dict)}"
            )

def save_split_data(sub_data, file_name):
    with open(file_name, 'w') as f:
        json.dump(sub_data, f)


if __name__ == "__main__":
    start_time = time.time()
    data_path = "/home/dufei.df/huaniu_workspace/Data/image_data_vcg_joy_smalltest.parquet"
    num_total_rank = 32
    res_path = f"/home/dufei.df/huaniu_workspace/Data/image_data_vcg_joy_smalltest_1024_bs2_node{num_total_rank}"
    os.makedirs(res_path, exist_ok=True)
    bucket_config = {
        # "144p": {0: (1.0, 80), 2: (1.0, 20), 4: ((1.0, 0.5), 10)},
        # # ---
        # "256": {0: (0.4, 50), 2: (0.5, 6), 4: ((0.5, 0.33), 3)},
        # "240p": {2: (1.0, 6), 4: (1.0, 4)},
        "1024": {0: (1.0, 2)}
        # "240p": {0: (1, 4)} #, 2: (1, 6), 4: (1, 3), 6: (1, 3)},
    }
    data = read_file(data_path)
    if 'video' not in os.path.basename(data_path):
        data.loc[:, 'duration'] = 0
    data["id"] = np.arange(len(data))
    sampler = Sampler(data, bucket_config, num_replicas=num_total_rank, shuffle=True, seed=1024, drop_last=True, verbose=True, num_bucket_build_workers=16)
    bucket_id_to_index = sampler.get_split()
    
    data.drop(['id', 'height', 'width', 'duration'], axis=1, inplace=True)
    
    print("save split files.")
    data.loc[:, 'bucket_id'] = "test"
    
    for i in tqdm(range(num_total_rank)):
        file_name = os.path.join(res_path, f"split_data_{i}.parquet")
        
        all_index = []
        for key in bucket_id_to_index[i]:
            sub_df = data.loc[bucket_id_to_index[i][key], "bucket_id"] = key
            all_index.extend(bucket_id_to_index[i][key])
        
        res_df = data.iloc[all_index]
        res_df = res_df.reset_index(drop=True)
        res_df.to_parquet(file_name, index=False)
        
    with open(os.path.join(res_path, "bucket_config.json"), 'w') as f:
        json.dump(bucket_config, f)
    
    
    print("consume time: ", time.time() - start_time, "s")
    