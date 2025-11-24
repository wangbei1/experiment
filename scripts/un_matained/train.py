import os, sys
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pprint import pformat
import numpy as np
import random
import functools
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
# DDP
from torch.nn.parallel import DistributedDataParallel as DDP
# FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import wandb
from tqdm import tqdm

from vidgen.acceleration.checkpoint import set_grad_checkpoint
from vidgen.acceleration.parallel_states import get_data_parallel_group
from vidgen.datasets.dataloader import prepare_dataloader
from vidgen.registry import DATASETS, MODELS, SCHEDULERS, build_module
from vidgen.utils.ckpt_utils import load, model_gathering, model_sharding, record_model_param_shape, save
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
from vidgen.utils.train_utils import TemporalMaskGenerator, update_ema, SpatialMaskGenerator

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(cfg.get("seed", 1024))
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

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
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
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = build_module(cfg.get("text_encoder", None), MODELS, device=device, dtype=dtype)
    if text_encoder is not None:
        text_encoder_output_dim = text_encoder.output_dim
        text_encoder_model_max_length = text_encoder.model_max_length
    else:
        text_encoder_output_dim = cfg.get("text_encoder_output_dim", 4096)
        text_encoder_model_max_length = cfg.get("text_encoder_model_max_length", 300)

    # == build vae ==
    vae = build_module(cfg.get("vae", None), MODELS)
    if vae is not None:
        vae = vae.to(device, dtype).eval()
        input_size = (dataset.num_frames, *dataset.image_size)
        latent_size = vae.get_latent_size(input_size)
        vae_out_channels = vae.out_channels
    else:
        latent_size = (None, None, None)
        vae_out_channels = cfg.get("vae_out_channels", 4)

    # == build diffusion model ==
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae_out_channels,
            caption_channels=text_encoder_output_dim,
            model_max_length=text_encoder_model_max_length,
            enable_sequence_parallelism=cfg.get("sp_size", 1) > 1,
        )
        # .to(device, dtype)
        .to(device) # model is suggested to keep fp32 dtype with AMP and FSDP. Due to: https://github.com/huggingface/accelerate/issues/2624
    )
    patch_size = model.patch_size
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "[Diffusion] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
    # == build ema for diffusion model ==
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema.eval()
    update_ema(ema, model, decay=0)
    
    # == setup loss function, build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # == setup optimizer ==
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), # params_to_optimize
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )

    warmup_steps = cfg.get("warmup_steps", None)

    if warmup_steps is None:
        lr_scheduler = None
    else:
        lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=cfg.get("warmup_steps"))

    # =======================================================
    # 4. distributed training preparation 
    # =======================================================
    logger.info("Preparing for distributed training...")
    torch.set_default_dtype(dtype)
    model = model.to(torch.cuda.current_device())
    mode = cfg.get('mode', 'DDP')
    local_rank = dist.get_rank() % torch.cuda.device_count()
    if mode == 'DDP':
        model = DDP(model, 
                device_ids             = [local_rank],
                output_device          = local_rank,
                find_unused_parameters = True)
    elif mode == 'FSDP':
        fpSixteen = MixedPrecision(
            param_dtype=dtype,
            # Gradient communication precision.
            reduce_dtype=torch.float,
            # Buffer precision.
            buffer_dtype=dtype,
        )

        my_size_based_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=100) #值越小，显存越大，速度稍微快一些
        
        model = FSDP(model, mixed_precision=fpSixteen, auto_wrap_policy=my_size_based_auto_wrap_policy, device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True)
        ema = FSDP(ema, mixed_precision=fpSixteen, auto_wrap_policy=my_size_based_auto_wrap_policy, device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True)
    else:
        print("Other training mode besides DDP and FSDP is not supported now.")
        sys.exit(0)
    

    # == additional preparation ==    
    if cfg.get("temporal_mask_ratios", None) is not None:
        xt_mask_generator = TemporalMaskGenerator(cfg.temporal_mask_ratios)
    if cfg.get("spatial_mask_ratio", None) is not None:
        xs_mask_generator = SpatialMaskGenerator(cfg.spatial_mask_ratio, patch_size)
            
    # reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")

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
        ret = load(
            cfg.load,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            sampler=None if cfg.get("start_from_scratch", False) else sampler,
        )
        if not cfg.get("start_from_scratch", False):
            start_epoch, start_step = ret
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)

    # =======================================================
    # 5. training loop
    # =======================================================
    dist.barrier()
    timers = {}
    timer_keys = [
        "move_data",
        "encode_video",
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
                timer_list = []
                with timers["move_data"] as move_data_t:
                    x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                    y = batch.pop("text")
                if record_time:
                    timer_list.append(move_data_t)

                # == visual and text encoding ==
                with torch.no_grad():
                    # Prepare visual inputs
                    with timers["encode_video"] as encode_video_t:
                        if cfg.get("load_video_features", False):
                            x = x.to(device, dtype)
                        else:
                            x = vae.encode(x)  # [B, C, T, H/P, W/P]
                    if record_time:
                        timer_list.append(encode_video_t)
                    # Prepare text inputs
                    with timers["encode_text"] as encode_text_t:
                        if cfg.get("load_text_features", False):
                            model_args = {"y": y.to(device, dtype)}
                            mask = batch.pop("mask")
                            if isinstance(mask, torch.Tensor):
                                mask = mask.to(device, dtype)
                            model_args["mask"] = mask
                        else:
                            model_args = text_encoder.encode(y)
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

                # == diffusion loss computation ==
                with timers["diffusion"] as loss_t:
                    loss_dict = scheduler.training_losses(model, x, model_args, xt_mask=xt_mask, xs_mask=xs_mask,
                                                          mae_loss_coef=cfg.get("mae_loss_coef", None), 
                                                          unpatchify_loss=cfg.get("unpatchify_loss", False))
                if record_time:
                    timer_list.append(loss_t)

                # == backward & update ==
                with timers["backward"] as backward_t:
                    loss = loss_dict["loss"].mean()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if cfg.get('grad_clip', None) is not None:
                        if mode == 'FSDP':
                            model.clip_grad_norm_(cfg.get('grad_clip'))
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('grad_clip'))

                    # update learning rate
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                if record_time:
                    timer_list.append(backward_t)

                # == update EMA ==
                with timers["update_ema"] as ema_t:
                    update_ema(ema, model, decay=cfg.get("ema_decay", 0.9999))
                if record_time:
                    timer_list.append(ema_t)

                # == update log info ==
                with timers["reduce_loss"] as reduce_loss_t:
                    all_reduce_mean(loss)
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
                if dist.get_rank() == 0 and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    avg_loss = running_loss / log_step
                    # progress bar
                    pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    # tensorboard
                    tb_writer.add_scalar("loss", loss.item(), global_step)
                    if "mae_loss" in loss_dict:
                        rf_avg_loss = rf_running_loss / log_step
                        mae_avg_loss = mae_running_loss / log_step
                        pbar.set_postfix({"loss": avg_loss, "rf_loss": rf_avg_loss, "mae_loss": mae_avg_loss, "step": step, "global_step": global_step})
                        tb_writer.add_scalar("rf_loss", rf_loss.item(), global_step)
                        tb_writer.add_scalar("mae_loss", mae_loss.item(), global_step)

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
                                    "debug/encode_video_time": encode_video_t.elapsed_time,
                                    "debug/encode_text_time": encode_text_t.elapsed_time,
                                    "debug/temporal_mask_time": mask_t.elapsed_time,
                                    "debug/spatial_mask_time": mask_s.elapsed_time,
                                    "debug/diffusion_time": loss_t.elapsed_time,
                                    "debug/backward_time": backward_t.elapsed_time,
                                    "debug/update_ema_time": ema_t.elapsed_time,
                                    "debug/reduce_loss_time": reduce_loss_t.elapsed_time,
                                }
                            )
                        wandb.log(wandb_dict, step=global_step)

                    running_loss = 0.0
                    rf_running_loss = 0.0
                    mae_running_loss = 0.0
                    log_step = 0

                # == checkpoint saving ==
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    save_dir = save(
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
                    )
                    logger.info(
                        "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                        epoch,
                        step + 1,
                        global_step + 1,
                        save_dir,
                    )
                if record_time:
                    log_str = f"Rank {dist.get_rank()} | Epoch {epoch} | Step {step} | "
                    for timer in timer_list:
                        log_str += f"{timer.name}: {timer.elapsed_time:.3f}s | "
                    print(log_str)

        sampler.reset()
        start_step = 0


if __name__ == "__main__":
    main()
