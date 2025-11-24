import os, sys
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pprint import pformat
import numpy as np
import random
import math
from PIL import Image
import functools
import gc
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.device_mesh import init_device_mesh
import wandb
from tqdm import tqdm
from vidgen.acceleration.checkpoint import set_grad_checkpoint
from vidgen.acceleration.parallel_states import get_data_parallel_group
from vidgen.datasets.dataloader import prepare_dataloader
from vidgen.registry import DATASETS, MODELS, SCHEDULERS, build_module
from vidgen.utils.ckpt_utils import async_save, sharded_load, load_checkpoint
from vidgen.utils.config_utils import define_experiment_workspace, parse_configs, save_training_config
from vidgen.utils.lr_scheduler import LinearWarmupLR
from vidgen.utils.misc import (
    Timer,
    all_reduce_mean,
    create_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
)
from vidgen.utils.train_utils import TemporalMaskGenerator, update_ema, SpatialMaskGenerator, CubeSpatialMaskGenerator
from vidgen.models.text_encoder import WanX21T5Encoder
from vidgen.models.wanx.clip import clip_xlm_roberta_vit_h_14
from vidgen.models.wanx.tokenizers import HuggingfaceTokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def encode_prompt(
    prompt,
    control_imgs,
    text_encoder,
    clip_model,
    max_seq_len):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    context = text_encoder(prompt)
    clip_feat = clip_model.visual(control_imgs)

    return dict(context=context, clip_feat=clip_feat, max_seq_len=max_seq_len)

def build_eval_dataset(cfg, resolution, num_frames, batch_size, data_parallel_group):
    bucket_config = {resolution: {num_frames: (1.0, batch_size)}}
    dataset = build_module(cfg.eval_dataset, DATASETS)
    dataloader_args = dict(
        dataset=dataset,
        batch_size=None,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        process_group=data_parallel_group,
    )
    dataloader, sampler = prepare_dataloader(bucket_config=bucket_config, **dataloader_args)
    num_steps_per_epoch = len(dataloader)
    return dataloader, num_steps_per_epoch

class CLIP:
    def __init__(self, name, dtype, device, checkpoint_path, tokenizer_path):
        self.name = name
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device
        )
        self.model = self.model.eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=self.model.max_text_len - 2,
            clean='whitespace'
        )
    
    def visual(self, videos):
        # preprocess
        size = (self.model.image_size, ) * 2
        videos = F.interpolate(videos, size=size, mode='bicubic', align_corners=False)
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))

        # forward
        with amp.autocast("cuda", dtype=self.dtype):
            out = self.model.visual(videos, use_31_block=True)
            return out

def create_mask(control_imgs,
                num_frames,
                ):
    
    control_img_h, control_img_w = control_imgs.shape[-2:]
    
    lat_h, lat_w = control_img_h // 8, control_img_w // 8
    # 构建 mask
    msk = torch.ones(control_imgs.shape[0], num_frames, lat_h, lat_w, device=control_imgs.device, dtype=control_imgs.dtype)
    msk[:, 1:] = 0
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
        dim=1
    )
    msk = msk.view(control_imgs.shape[0], -1, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)
    
    return msk


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=True)
    
    record_time = cfg.get("record_time", False)

    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    # == init distributed training ==
    # NOTE: A very large timeout is set to avoid some processes exit early
    # dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))    
    _rank = int(os.environ["RANK"])
    _world_size = int(os.environ["WORLD_SIZE"])
    
    #初始化一个process_group，可以避免使用torch2.5的device_mesh_flatten
    dist.init_process_group("cpu:gloo,cuda:nccl", timeout=timedelta(hours=24))
    
    device_num = torch.cuda.device_count()
    
    mesh_size = (_world_size // device_num, device_num)
    mesh_dims = ("rep", "shard")
    device_mesh = init_device_mesh("cuda", mesh_size, mesh_dim_names=mesh_dims)
    # 获取当前 rank 在 mesh 中的位置
    mesh_coord = device_mesh.get_coordinate()  # 不需要传入参数
    print(f"mesh_coord on current (global)rank-{dist.get_rank()}:", mesh_coord)
    is_first_in_group = mesh_coord[1] == 0  # mesh_coord 是一个元组 (rep_idx, shard_idx)
    # device_mesh._flatten("dp")
    
    set_seed(cfg.get("seed", 1024))
    torch.cuda.set_device(dist.get_rank() % device_num)
    device = torch.cuda.current_device()

    # == init exp_dir ==
    exp_name, exp_dir = define_experiment_workspace(cfg)
    
    dist.barrier()
    if dist.get_rank() == 0: # only master should do makedirs
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    dist.barrier()

    # == init logger, tensorboard & wandb ==
    logger = create_logger(exp_dir)
    logger.info("Experiment directory created at %s", exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    if dist.get_rank() == 0:
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(project="Open-Sora", name=exp_name, config=cfg.to_dict(), dir="./outputs/wandb")
            
    # training_acc initialize
    from astra.initialize import initialize
    from astra.accelerate import accelerate
    from astra.parallel import dist as para_dist
    
    sp_degree = cfg.get("sp_degree", 1)
    enable_sp = True if sp_degree > 1 else False
    initialize(sp_parallel_size=sp_degree)
    # =======================
    
    monitor = cfg.get("monitor", False)
    
    data_parallel_group = get_data_parallel_group()
    if enable_sp:
        data_parallel_group = para_dist.parallel_state.get_data_parallel_group()
    
    # ======================================================
    # 2. build model
    # ======================================================
    logger.info("Building models...")
    if is_first_in_group:
        # rank 0 直接在 CPU 上创建
        # 但是此时CPU内存不是最优的。model创建及初始化占用一倍模型参数、load_checkpoint再一倍。
        # 如果以后，万一，需要优化这个地方的内存使用，可以考虑分块加载的方式，比如调用ckpt_utils.load_model_by_layer。
        # 不要考虑在rank0上搞meta_device，然后再进行有pretrain部分参数的加载。这样会需要特殊处理额外自定义参数的materialize
        model = build_module(cfg.model, MODELS)
        # 在（分组的）rank0上显式调用init_weight，进行实例化(materialization)。不要让该方法在model的__init__被调用。
        model.init_weights() # 如果有特殊的init_weight逻辑，比如类似controlnet的zero init，记得在init_weight里面去实现！
        print(f"random init weight for model.")
        if cfg.model.from_pretrained is not None:
            load_checkpoint(model, cfg.model.from_pretrained)
            gc.collect()
            print(f"load wanx model from {cfg.model.from_pretrained} at CPU")
        param_init_fn = None
    else:
        # 其他情况都在 meta device 上创建
        with torch.device("meta"):
            model = build_module(cfg.model, MODELS)
        param_init_fn = lambda module: module.to_empty(device=torch.cuda.current_device(), recurse=False)
        
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "[Diffusion] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
    
    # parallelize model
    model = accelerate("wanx2_1_i2v", model)
        
    # == setup loss function, build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # =======================================================
    # 3. distributed training preparation 
    # =======================================================
    logger.info("Preparing for distributed training...")
    torch.set_default_dtype(dtype)
    # model = model.to(torch.cuda.current_device())
    mode = cfg.get('mode', 'FSDP')
    if mode == 'FSDP':
        fsdp_config = dict(
            auto_wrap_policy=functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in model.blocks
            ),
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=torch.float32,
                buffer_dtype=dtype,
            ),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            device_mesh=device_mesh["rep", "shard"],
            param_init_fn=param_init_fn
        )
        fsdp_config.update(sync_module_states=True) # 添加这个配置确保模型从rank-0同步到同shard组的其他rank
        model = FSDP(model, **fsdp_config)
    else:
        print("Other training mode besides DDP and FSDP is not supported now.")
        sys.exit(0)
        
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Finish warp model for distributed training")
        # == build text-encoder and vae ==
    text_encoder = WanX21T5Encoder(
        name=cfg.t5.name,
        text_len=cfg.t5.text_len,
        dtype=cfg.t5.dtype,
        device=device,
        checkpoint_path=cfg.t5.checkpoint_path,
        tokenizer_path=cfg.t5.tokenizer_path
    )
    if os.getenv('ENABLE_COMPILE', 'True').lower() == 'true':
        logger.info("Compiling WanX text_encoder")
        text_encoder.model = torch.compile(text_encoder.model)
    if mode == 'FSDP':
        # FSDP text_encoder
        fsdp_config.update(sync_module_states=False)
        fsdp_config.update(
            auto_wrap_policy=functools.partial(
                lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in text_encoder.model.blocks,
            )
        )
        fsdp_config.update(param_init_fn=None)
        text_encoder.model = FSDP(text_encoder.model, **fsdp_config).requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Finish build text encoder")
    
    # == build vae ==
    vae = build_module(cfg.get("vae", None), MODELS)
    vae.model = vae.model.to(device=device, dtype=dtype).eval()
    if os.getenv('ENABLE_COMPILE', 'True').lower() == 'true':
        vae.model = torch.compile(vae.model)
    
    clip_model = CLIP(
        name=cfg.clip_model,
        dtype=torch.float16,
        device=device,
        checkpoint_path=cfg.clip_checkpoint,
        tokenizer_path=cfg.clip_tokenizer
    )
    if os.getenv('ENABLE_COMPILE', 'True').lower() == 'true':
        clip_model.model = torch.compile(clip_model.model)
    gc.collect()
    torch.cuda.empty_cache()

    # TODO: wanx 不支持，暂时注释掉
    # vae.enable_tiling()
    # vae.enable_slicing()
    # == build ema for diffusion model ==
    with torch.device("meta"):
        ema = build_module(cfg.model, MODELS).eval() 
    if mode == 'FSDP':
        fsdp_config.update(
            auto_wrap_policy=functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in ema.blocks
            ),
            param_init_fn = lambda mod: mod.to_empty(device='cuda', recurse=False),
            sync_module_states=False
        )
        ema = FSDP(ema, **fsdp_config).requires_grad_(False)
    update_ema(ema, model, decay=0, initialize=True)
    logger.info("Finish build ema")

    # == setup loss function, build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # == setup optimizer ==
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), # params_to_optimize
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )
    logger.info("Finish setup optimizer")

    # == setup lr_scheduler ==
    warmup_steps = cfg.get("warmup_steps", None)

    if warmup_steps is None:
        lr_scheduler = None
    else:
        lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=cfg.get("warmup_steps"))

    prompt_uncond_prob = cfg.get("prompt_uncond_prob", 0.1)
    # == additional preparation ==    
    if cfg.get("temporal_mask_ratios", None) is not None:
        xt_mask_generator = TemporalMaskGenerator(cfg.temporal_mask_ratios)
    if cfg.get("spatial_mask_ratio", None) is not None:
        patch_size = model.config.patch_size
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)
        if cfg.get("is_cube", False):
            xs_mask_generator = CubeSpatialMaskGenerator(cfg.spatial_mask_ratio, patch_size, cfg.cube_size)
        else:
            xs_mask_generator = SpatialMaskGenerator(cfg.spatial_mask_ratio, patch_size)
            
    # reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(torch.float)

    # ======================================================
    # 4. build dataset and dataloader
    # ======================================================
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Building dataset...")
    # == build dataset ==
    cfg.dataset['rank'] = data_parallel_group.rank()
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=data_parallel_group,
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)
    
    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    start_epoch = start_step = log_step = acc_step = 0
    running_loss = 0.0
    mae_running_loss = 0.0
    rf_running_loss = 0.0
    
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)
    
    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        ret = sharded_load(
            cfg.load,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            sampler=None if cfg.get("start_from_scratch", False) else sampler
        )
        if not cfg.get("start_from_scratch", False):
            start_epoch, start_step = ret
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # =======================================================
    # 5. training loop
    # =======================================================
    dist.barrier()
    timers = {}
    timer_keys = [
        "move_data",
        "encode_vae",
        "encode_text",
        "temporal_mask",
        "spatial_mask",
        "diffusion",
        "backward",
        "update_ema",
        "reduce_loss",
    ]
    for key in timer_keys:
        if record_time:
            timers[key] = Timer(key, dist=dist)
        else:
            timers[key] = nullcontext()
    
    async_save_handle = None
    smoothing_running_loss_avg_dict = {"video": None, "image": None}
    
    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)
        
        model.train()
        
        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not dist.get_rank() == 0,
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                optimizer.zero_grad()
                if enable_sp:
                    # invalid batch, drop current step
                    valid = batch["valid"]
                    if all(valid):
                        batch_valid = torch.ones(1).bool().to(device)
                    else:
                        batch_valid = torch.zeros(1).bool().to(device)
                        
                    from training_acc.comm.split_gather import _gather
                    batch_valid = _gather(batch_valid)
                    if not all(batch_valid):
                        from training_acc.logger import logger as acc_logger
                        acc_logger.warning(f"[rank {dist.get_rank()}] step:{step}. data invalid, skip!!!")
                        continue
                timer_list = []
                with timers["move_data"] as move_data_t:
                    x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                    y = batch.pop("text")
                    
                if record_time:
                    timer_list.append(move_data_t)
                # == visual and text encoding ==
                
                # 从视频中获取到参考图像                                
                control_imgs = x[:, :, [0]]
                num_frames = x.shape[2]
                mask = create_mask(control_imgs, num_frames)
                
                with torch.no_grad():
                    img_mask_latent = vae.encode(torch.cat([control_imgs, control_imgs.new_zeros(control_imgs.shape[0], control_imgs.shape[1], num_frames-1, control_imgs.shape[3], control_imgs.shape[4])], dim=2))
                    img_mask_latent = torch.stack(img_mask_latent)
                    img_mask_latent = torch.concat([mask, img_mask_latent], dim=1)
                
                with timers["encode_vae"] as encode_vae_t:
                    with torch.no_grad():
                        # Prepare visual inputs
                        x_shape_record = x.shape
                        if cfg.get("load_video_features", False):
                            x = x.to(device, dtype)
                        else:
                            x = vae.encode(x)
                            x = torch.stack(x)
                        
                if record_time:
                    timer_list.append(encode_vae_t)
                    
                with timers["encode_text"] as encode_text_t:
                    with torch.no_grad():
                        if cfg.get("load_text_features", False):
                            model_args = {"y": y.to(device, dtype)}
                            mask = batch.pop("mask")
                            if isinstance(mask, torch.Tensor):
                                mask = mask.to(device, dtype)
                            model_args["mask"] = mask
                        else:
                            drop_ids = np.random.rand(len(y)) < prompt_uncond_prob
                            y = ["" if drop_ids[i] else y[i] for i in range(len(y))] #置空
                            model_args = encode_prompt(
                                prompt=y,
                                control_imgs=control_imgs.squeeze(2),
                                text_encoder=text_encoder,
                                clip_model=clip_model,
                                max_seq_len=cfg.max_seq_len)
                        
                if record_time:
                    timer_list.append(encode_text_t)

                # == temporal mask ==
                with timers["temporal_mask"] as mask_xt:
                    xt_mask = None
                    if cfg.get("temporal_mask_ratios", None) is not None:
                        xt_mask = xt_mask_generator.get_masks(x)
                        model_args["xt_mask"] = xt_mask
                if record_time:
                    timer_list.append(mask_xt)
                    
                # == spatial mask ==
                with timers["spatial_mask"] as mask_xs:
                    xs_mask = None
                    if cfg.get("spatial_mask_ratio", None) is not None:
                        xs_mask = xs_mask_generator.get_mask(x)
                        model_args["xs_mask"] = xs_mask
                if record_time:
                    timer_list.append(mask_xs)
                    
                # == video meta info ==
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        model_args[k] = v.to(device, dtype)

                if enable_sp:
                    model_args["enable_sp"] = True
                    
                # == diffusion loss computation ==
                model_args["img_mask_latent"] = img_mask_latent
                with timers["diffusion"] as loss_t:
                    loss_dict = scheduler.training_losses(model,
                                                          x,
                                                          model_args,
                                                          xt_mask=xt_mask,
                                                          xs_mask=xs_mask,
                                                          mae_loss_coef=cfg.get("mae_loss_coef", None), 
                                                          unpatchify_loss=cfg.get("unpatchify_loss", False), 
                                                          mode='i2v')
                if record_time:
                    timer_list.append(loss_t)

                # == backward & update ==
                with timers["backward"] as backward_t:
                    loss = loss_dict["loss"].mean()
                    loss.backward()

                    if cfg.get('grad_clip', None) is not None:
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('grad_clip'))
                        model.clip_grad_norm_(cfg.get('grad_clip'))
                    
                    all_reduce_mean(loss)
                    is_model_update = False
                    s_key = 'image' if x.shape[2] == 1 else 'video'
                    
                    if smoothing_running_loss_avg_dict[s_key] is None or loss.item() < 5 * smoothing_running_loss_avg_dict[s_key]:
                        
                        optimizer.step()
                        
                        is_model_update = True
                        # update learning rate
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                            
                        smoothing_running_loss_avg_dict[s_key] = (smoothing_running_loss_avg_dict[s_key] * 0.9 + loss.item() * 0.1) if smoothing_running_loss_avg_dict[s_key] is not None else loss.item()
                    else:
                        logger.info(f"skip update model because of large loss {loss.item()}, smoothing_running_loss_avg {smoothing_running_loss_avg_dict[s_key]}")
                if record_time:
                    timer_list.append(backward_t)

                # == update EMA ==
                with timers["update_ema"] as ema_t:
                    if is_model_update:
                        update_ema(ema, model, decay=cfg.get("ema_decay", 0.9999))
                if record_time:
                    timer_list.append(ema_t)

                # == update log info ==
                with timers["reduce_loss"] as reduce_loss_t:
                    # all_reduce_mean(loss)
                    running_loss += loss.item()
                    if "mae_loss" in loss_dict:
                        mae_loss = loss_dict["mae_loss"].mean()
                        all_reduce_mean(mae_loss)
                        rf_loss = loss_dict["rf_loss"].mean()
                        all_reduce_mean(rf_loss)
                        mae_running_loss += mae_loss.item()
                        rf_running_loss += rf_loss.item()
                        
                    global_step = epoch * num_steps_per_epoch + step
                    log_step += 1
                    acc_step += 1
                if record_time:
                    timer_list.append(reduce_loss_t)

                # == logging ==
                if dist.get_rank() == 0 and (global_step + 1) % cfg.get("log_every", 1) == 0 and log_step != 0:
                    avg_loss = running_loss / log_step
                    # tensorboard
                    if is_model_update:
                        tb_writer.add_scalar("loss", loss.item(), global_step)
                    if "mae_loss" in loss_dict:
                        rf_avg_loss = rf_running_loss / log_step
                        mae_avg_loss = mae_running_loss / log_step
                        pbar.set_postfix({"loss": avg_loss, "smooth_loss": smoothing_running_loss_avg_dict[s_key], "rf_loss": rf_avg_loss, "mae_loss": mae_avg_loss, "lr": optimizer.param_groups[0]["lr"], "step": step, "global_step": global_step})
                        if is_model_update:
                            tb_writer.add_scalar("rf_loss", rf_loss.item(), global_step)
                            tb_writer.add_scalar("mae_loss", mae_loss.item(), global_step)
                    else:
                        # progress bar
                        pbar.set_postfix({"loss": avg_loss, "smooth_loss": smoothing_running_loss_avg_dict[s_key], "lr": optimizer.param_groups[0]["lr"], "step": step, "global_step": global_step})
                        
                    # wandb
                    if cfg.get("wandb", False):
                        wandb_dict = {
                            "iter": global_step,
                            "acc_step": acc_step,
                            "epoch": epoch,
                            "loss": loss.item(),
                            "avg_loss": avg_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                        if "mae_loss" in loss_dict:
                            wandb_dict["rf_loss"] = rf_loss.item()
                            wandb_dict["mae_loss"] = mae_loss.item()
                        if record_time:
                            wandb_dict.update(
                                {
                                    "debug/move_data_time": move_data_t.elapsed_time,
                                    "debug/encode_time": encode_t.elapsed_time,
                                    "debug/mask_time": mask_t.elapsed_time,
                                    "debug/diffusion_time": loss_t.elapsed_time,
                                    "debug/backward_time": backward_t.elapsed_time,
                                    "debug/update_ema_time": ema_t.elapsed_time,
                                    "debug/reduce_loss_time": reduce_loss_t.elapsed_time,
                                }
                            )
                        wandb.log(wandb_dict, step=global_step)

                    running_loss = 0.0
                    if "mae_loss" in loss_dict:
                        rf_running_loss = 0.0
                        mae_running_loss = 0.0
                    log_step = 0
                    
                if monitor:
                    from training_acc.utils import torch_cuda_mem_inspect
                    torch_cuda_mem_inspect(int(os.environ["LOCAL_RANK"]))
                    # print(f"rank {dist.get_rank()} | memory allocated {torch.cuda.memory_allocated() / 1024 ** 3:.02f} GB  all reserved memory {torch.cuda.max_memory_allocated() / 1024 ** 3:.02f} GB for x_shape {x_shape_record}")

                # == checkpoint saving ==
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    torch.cuda.empty_cache()
                    save_dir = os.path.join(exp_dir, f"epoch{epoch}-global_step{global_step}")
                    gc.collect()
                    dist.barrier()
                    async_save_handle = async_save(
                        exp_dir,
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        sampler=sampler,
                        epoch=epoch,
                        step=step + 1,
                        global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                        device_mesh=device_mesh,
                        last_dcp_handles=async_save_handle
                    )
                    
                    logger.info(
                        "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                        epoch,
                        step + 1,
                        global_step + 1,
                        save_dir,
                    )
                    torch.cuda.empty_cache()
                if record_time:
                    log_str = f"Rank {dist.get_rank()} | Epoch {epoch} | Step {step} | "
                    for timer in timer_list:
                        log_str += f"{timer.name}: {timer.elapsed_time:.3f}s | "
                    print(log_str)
                                                            
                eval_every = cfg.get("eval_every", 0)
                if eval_every > 0 and (global_step + 1) % cfg.get("eval_every", 1) == 0:
                    model.eval()
                    evaluation_losses = {}
                    eval_bucket_config = cfg.eval_bucket_config
                    val_loss = 0.
                    val_num = 0
                    for i, eval_res in enumerate(eval_bucket_config):
                        t_bucket = eval_bucket_config[eval_res]
                        for eval_num_frames, (_, eval_batch_size) in t_bucket.items():
                            if eval_batch_size is None:
                                continue
                            logger.info("Evaluating resolution: %s, num_frames: %s", eval_res, eval_num_frames)
                            eval_dataloader, eval_num_steps_per_epoch = build_eval_dataset(cfg, eval_res, eval_num_frames, eval_batch_size, data_parallel_group)
                            if eval_num_steps_per_epoch == 0:
                                logger.warning("No data for resolution: %s, num_frames: %s", eval_res, eval_num_frames)
                                continue
                            
                            eval_bucket_loss = torch.tensor(0., device=device)
                            eval_num_samples = torch.tensor(0, device=device)
                            eval_dataloader_iter = iter(eval_dataloader)
                            for _ in tqdm(range(eval_num_steps_per_epoch), disable=not dist.get_rank() == 0, desc=f"res: {eval_res}, num_frames: {eval_num_frames}"):
                                batch = next(eval_dataloader_iter)
                                x = batch.pop("video").to(device, dtype)
                                y = batch.pop("text")
                                with torch.no_grad():
                                    x = vae.encode(x)
                                    x = torch.stack(x)
                                    
                                    model_args = encode_prompt(
                                        prompt=y,
                                        neg_prompt=cfg.sample_neg_prompt,
                                        text_encoder=text_encoder,
                                        max_seq_len=cfg.max_seq_len)
                            
                                    # == video meta info ==
                                    for k, v in batch.items():
                                        model_args[k] = v.to(device, dtype)
                                    
                                    for t in torch.linspace(0, scheduler.num_timesteps, cfg.get("num_eval_timesteps", 10) + 2)[1:-1]:
                                        # == diffusion loss computation ==
                                        timestep = torch.tensor([t] * x.shape[0], device=device, dtype=dtype)
                                        # with torch.autocast(device_type="cuda"):
                                        loss_dict = scheduler.training_losses(model, x, model_args, t=timestep)
                                        losses = loss_dict["loss"].detach()  # (batch_size)
                                        eval_num_samples += x.shape[0]
                                        eval_bucket_loss += losses.sum()
                                        # print(t, " ", losses.sum(), " ", bucket_loss)
                                        # print("ttttt: ", t, losses)
                            
                            dist.reduce(tensor=eval_bucket_loss, dst=0, op=dist.ReduceOp.SUM)
                            dist.reduce(tensor=eval_num_samples, dst=0, op=dist.ReduceOp.SUM)
                                
                            if dist.get_rank() == 0:
                                bucket_loss_avg = eval_bucket_loss / eval_num_samples if torch.sum(eval_num_samples) else torch.tensor(0., device=device)
                                logger.info("Global step: %s, validation losses for resolution: %s, num_frames: %s, loss: %s\n",
                                    global_step + 1, eval_res, eval_num_frames, bucket_loss_avg)
                                tb_writer.add_scalar(f"val_loss/{eval_res}/{eval_num_frames}", bucket_loss_avg, global_step)
                                # update total loss and number
                                val_loss += eval_bucket_loss
                                val_num += eval_num_samples            
                    
                    # log final val loss
                    if dist.get_rank() == 0:
                        val_loss_avg = val_loss/val_num
                        logger.info(f"Validation loss at {global_step} steps: {val_loss_avg}")
                        tb_writer.add_scalar(f"val_loss/total", val_loss_avg, global_step)
                    torch.cuda.empty_cache()
                    model.train()

        sampler.reset()
        start_step = 0

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
