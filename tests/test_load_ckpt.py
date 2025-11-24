import os
import sys
from datetime import timedelta
import numpy as np
import random
import functools
import gc
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.device_mesh import init_device_mesh
from vidgen.acceleration.checkpoint import set_grad_checkpoint
from vidgen.acceleration.parallel_states import get_data_parallel_group
from vidgen.registry import MODELS, build_module
from vidgen.utils.misc import (
    get_memory_info,
    to_torch_dtype
)
from vidgen.utils.config_utils import parse_configs
import logging

def get_logger():
    return logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def log_memory_usage(rank):
    allocated = torch.cuda.memory_allocated(rank) / 1024**2
    reserved = torch.cuda.memory_reserved(rank) / 1024**2
    print(f"Rank {rank} - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")

def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=True)
    
    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == init distributed training ==
    _rank = int(os.environ["RANK"])
    _world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group("cpu:gloo,cuda:nccl", timeout=timedelta(hours=24))
    local_rank = dist.get_rank() % torch.cuda.device_count()
    
    device_num = torch.cuda.device_count()
    
    mesh_size = (_world_size // device_num, device_num)
    mesh_dims = ("rep", "shard")
    device_mesh = init_device_mesh("cuda", mesh_size, mesh_dim_names=mesh_dims)
    
    set_seed(cfg.get("seed", 1024))
    torch.cuda.set_device(dist.get_rank() % device_num)
    
    logger = get_logger()
    logger.info("Building models...")
    
    get_memory_info(msg="Before Building models...")
    with torch.device("meta"):
        model = build_module(cfg.model, MODELS) # at CPU model
    get_memory_info(msg="After Creating models...")
    
    if dist.get_rank() == 0:
        checkpoint = torch.load(cfg.model.from_pretrained, map_location='cpu', mmap=True)
        model.load_state_dict(checkpoint, assign=True)
        param_init_fn = None
        get_memory_info(msg="After Loading models...")
        log_memory_usage(0)
        log_memory_usage(7)
    else:
        param_init_fn = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False)
        
    # print(f"\n=== Parameter state model_cpu ===")
    # for name, param in model.named_parameters():
    #     print(f"{name}: shape {param.shape}")
    
    fsdp_config = dict(
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.blocks
        ),
        device_id=torch.device('cuda'),
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=dtype,
            _module_classes_to_ignore=(nn.LayerNorm, nn.RMSNorm)
        ),
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        device_mesh=device_mesh["rep", "shard"],
        param_init_fn = param_init_fn
    )
    fsdp_config.update(sync_module_states=True)
    model_fsdp = FSDP(model, **fsdp_config).eval()
    get_memory_info(msg="After FSDP...")
    log_memory_usage(0)
    log_memory_usage(7)
    
    
    
    # print(f"\n=== Parameter state FSDP ===")
    # for name, param in model_fsdp.named_parameters():
    #     print(f"{name}: shape {param.shape}")
        
    # 构造符合实际数据结构的模拟输入
    sample_input = {
        "x": torch.randn((1, 16, 21, 90, 160), device='cuda', dtype=dtype),
        "t": torch.zeros(1, device='cuda', dtype=dtype),
        "context": [torch.randn((512, 4096), device='cuda', dtype=dtype)],
        "seq_len": 75600
    }
    
    with torch.no_grad():
        output_fsdp = model_fsdp(**sample_input)
    
    del model_fsdp, model
    torch.cuda.empty_cache()
    if dist.get_rank() == 7:
        with torch.device("cuda"):
            model = build_module(cfg.model, MODELS) 
            checkpoint = torch.load(cfg.model.from_pretrained, map_location='cpu')
            model.load_state_dict(checkpoint)
            model.to(dtype=dtype).eval()
            with torch.no_grad():
                output_model = model(**sample_input)
            diff = (output_model - output_fsdp).abs()
            print(f"Mean absolute diff: {diff.mean()}")
            print(f"Max absolute diff: {diff.max()}")
            print(f"Relative diff: {(diff / (output_fsdp.abs() + 1e-8)).mean()}")
            print(f"Are close: {torch.allclose(output_model, output_fsdp, rtol=1e-3, atol=1e-3)}")
    
    
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()