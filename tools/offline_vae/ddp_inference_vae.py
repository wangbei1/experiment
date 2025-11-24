import os
import math
import torch
import gzip
import shutil
import random
import argparse
import threading
import queue
import numpy as np
from tqdm import tqdm
from datetime import timedelta
import video_transforms
from bucket import SecondsBucket
import torchvision.transforms as transforms
from tools.offline_vae.datasets import VariableVideoTextPerRankDatasetVae
from tools.offline_vae.sampler import VariableVideoBatchPerRankSamplerVae
from tools.offline_vae.jingkai_vae import JINGKAI_VAE
from torch.utils.data import DataLoader
from vidgen.utils.misc import Timer, create_logger
import torch.distributed as dist
import concurrent.futures

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_seed_worker(seed):
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker

class AsyncSaver:
    def __init__(self, max_workers=4):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    def save_numpy(self, array, filename):
        # 实际的保存函数
        def _save():
            save_path = filename + '.npz'
            np.savez_compressed(save_path, data=array)
        
        # 将保存任务提交给线程池
        self.executor.submit(_save)
    
    def shutdown(self):
        self.executor.shutdown(wait=True)

def save_worker(save_queue, stop_event):
    global logger, global_rank
    while not stop_event.is_set():
        try:
            iter_num, rank, latent, filename = save_queue.get(timeout=0.1)
            # 定义保存路径
            if os.path.exists(f"/mnt/dufei.df/save_cache/{rank}_{iter_num-1}.pt.gz"):
                os.remove(f"/mnt/dufei.df/save_cache/{rank}_{iter_num-1}.pt.gz")
            latent = latent.clone()
            with gzip.GzipFile(f"/mnt/dufei.df/save_cache/{rank}_{iter_num}.pt.gz", "w") as f:
                torch.save(latent, f)
            shutil.copy(f"/mnt/dufei.df/save_cache/{rank}_{iter_num}.pt.gz", filename+".pt.gz")
            logger.info(f"Saved latent for iter {iter_num} to {filename+'.pt.gz'}")
            save_queue.task_done()
        except Exception as e:
            if not isinstance(e, queue.Empty):
                logger.info(f"Saved latent for iter {iter_num} to {filename+'.pt.gz'} failed")
            continue
        
def encode_video(vae, x):
    with torch.no_grad():
        h, w = x.shape[-2:]
        batch_slice_2d = max(2 ** 22 // h // w, 1)
        z = vae.encode(x, time_slice=32, batch_slice_2d=batch_slice_2d).latent_dist.parameters
    return z

def main(args):
    rank = int(os.environ['RANK'])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24), rank=rank, world_size=world_size)
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    seed = 42
    set_seed(seed)
    
    device = torch.cuda.current_device()
    num_processes = dist.get_world_size()
    
    global global_rank
    
    global_rank = dist.get_rank()
    is_main_process_bool = global_rank == 0
    
    exp_dir = args.exp_dir
    dist.barrier()
    if dist.get_rank() == 0: # only master should do makedirs
        os.makedirs(exp_dir, exist_ok=True)
    dist.barrier()
    
    global logger
    
    logger = create_logger(exp_dir)
    
    spatial_vae_path = "/mnt/dufei.df/FLUX.1-dev-diffusers/"
    vae_path = "/mnt/dufei.df/video_vae/checkpoint-stage4-final.pth"
    
    dataset = VariableVideoTextPerRankDatasetVae(
        data_path=args.data_path,
        sample_fps=16,
        rank=global_rank,
        transform_name="resize_crop",
    )
    
    sampler = VariableVideoBatchPerRankSamplerVae(
        dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=False,
        seed=seed,
        drop_last=True,
        verbose=True,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        worker_init_fn=get_seed_worker(seed),
        pin_memory=True,
        num_workers=8
    )
    
    num_steps_per_epoch = len(dataloader)
    
    vae_model = JINGKAI_VAE(
        pretrained_spatial_vae_path=spatial_vae_path,
        from_pretrained=vae_path,
        scaling_factor=0.9480883479118347,
        shift_factor=0.04052285850048065
    ).to("cuda", dtype=torch.bfloat16)
    
    vae_model.eval()
    
    timers = {}
    timer_keys = [
        "move_data",
        "encode_vae",
        "save",
    ]
    record_time = args.record_time
    for key in timer_keys:
        if record_time:
            timers[key] = Timer(key, dist=dist)
        else:
            timers[key] = nullcontext()
    
    saver = AsyncSaver(max_workers=4)
    
    sampler.set_epoch(0)
    dataloader_iter = iter(dataloader)
    
    logger.info("Beginning inference.")
    
    #这个主要是为了保证能够resume，resume的时候直接用这个就行
    sampler.set_start_index(args.start_step)
    
    save_queue = queue.Queue()
    stop_event = threading.Event()
    saver_thread = threading.Thread(target=save_worker, args=(save_queue, stop_event))
    saver_thread.start()
    
    with tqdm(
            enumerate(dataloader_iter, start=args.start_step),
            desc=f"Epoch {0}",
            disable=not dist.get_rank() == 0,
            initial=args.start_step,
            total=num_steps_per_epoch,
        ) as pbar:
        
        for step, batch in pbar:
            timer_list = []
            with timers["move_data"] as move_data_t:
                x = batch.pop("video").to(device, vae_model.dtype)  # [B, C, T, H, W]
                paths = batch.pop("path")
                
            if record_time:
                timer_list.append(move_data_t)
            
            with timers["encode_vae"] as encode_vae_t:
                latents = encode_video(vae_model, x).cpu()
                
            if record_time:
                timer_list.append(encode_vae_t)
            
            #对每个样本的特征进行异步保存
            with timers["save"] as save_t:
                for latent, fname in zip(latents, paths):
                    os.makedirs(os.path.dirname(fname), exist_ok=True)
                    save_queue.put((step, global_rank, latent, fname))
            
            if record_time:
                timer_list.append(save_t)
                
            if dist.get_rank() == 0:
                pbar.set_postfix({"step": step})
                logger.info("Step: %s \n", step)
            
            if record_time:
                log_str = f"Rank {dist.get_rank()} | Step {step} | "
                for timer in timer_list:
                    log_str += f"{timer.name}: {timer.elapsed_time:.3f}s | "
                print(log_str)
            
            dist.barrier()
        
        save_queue.join()

        # 停止保存线程
        stop_event.set()
        saver_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--exp_dir", default="outputs/inference_vae", type=str)
    parser.add_argument("--start_step", default=0, type=int)
    parser.add_argument("--record-time", action="store_true", default=False)
    
    
    args = parser.parse_args()
    main(args)