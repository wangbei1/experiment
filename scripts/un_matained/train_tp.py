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
# TP/SP/FSDP
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, Replicate
from torch.distributed._tensor.experimental import implicit_replication
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)
# FSDP2
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

import wandb
from tqdm import tqdm

from vidgen.acceleration.checkpoint import set_grad_checkpoint
from vidgen.acceleration.parallel_states import get_data_parallel_group
from vidgen.datasets.dataloader import prepare_dataloader
from vidgen.registry import DATASETS, MODELS, SCHEDULERS, build_module
from vidgen.utils.ckpt_utils import load, model_gathering, model_sharding, record_model_param_shape, save, sharded_save, sharded_load
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

from torch.distributed._tensor.experimental import implicit_replication


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parallelize_model(model, tp_mesh, dp_mesh, dtype):
    if tp_mesh is not None:
        for layer_id, st_blocks in enumerate(model.st_blocks):
            layer_tp_plan = {
                "scale_shift_table":PrepareModuleInput(
                    input_layouts=Replicate(),
                    desired_input_layouts=Replicate(),
                ),
                # "attention_norm": SequenceParallel(),
                # "attn": PrepareModuleInput(
                #     input_layouts=(Shard(1), None),
                #     desired_input_layouts=(Replicate(), None),
                # ),
                "attn.wq": ColwiseParallel(),
                "attn.wk": ColwiseParallel(),
                "attn.wv": ColwiseParallel(),
                "attn.proj": RowwiseParallel(),
                # "norm1": SequenceParallel(),
                "cross_attn.q_linear": ColwiseParallel(),
                "cross_attn.k_linear": ColwiseParallel(),
                "cross_attn.v_linear": ColwiseParallel(),
                "cross_attn.proj": RowwiseParallel(),
                # "norm2": SequenceParallel(),
                # "mlp": PrepareModuleInput(
                #     input_layouts=(Shard(1),),
                #     desired_input_layouts=(Replicate(),),
                # ),
                "mlp.fc1": ColwiseParallel(),
                # "mlp.drop1": SequenceParallel(),
                "mlp.fc2": RowwiseParallel(),
            }

            # Adjust attention module to use the local number of heads
            st_blocks.attn.num_heads = st_blocks.attn.num_heads // tp_mesh.size()
            st_blocks.cross_attn.num_heads = st_blocks.cross_attn.num_heads // tp_mesh.size()

            # Custom parallelization plan for the model
            parallelize_module(
                module=st_blocks,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan
            )
        # parallelize the first embedding and the last linear out projection
        model = parallelize_module(
            model,
            tp_mesh,
            {
                "x_embedder.proj": RowwiseParallel(
                    input_layouts=Replicate(),
                    # output_layouts=Shard(1),
                ),
                "t_embedder.mlp.0": RowwiseParallel(
                    input_layouts=Replicate(),
                ),
                "t_embedder.mlp.2": ColwiseParallel(
                    output_layouts=Replicate(),
                ),

                "fps_embedder.mlp.0": RowwiseParallel(
                    input_layouts=Replicate(),
                ),
                "fps_embedder.mlp.2": ColwiseParallel(
                    output_layouts=Replicate(),
                ),

                "t_block.1": RowwiseParallel(
                    input_layouts=Replicate(),
                    # output_layouts=Shard(1),
                ),

                "y_embedder.mlp.fc1": RowwiseParallel(),
                "y_embedder.mlp.fc2": ColwiseParallel(),
                "final_layer.scale_shift_table":PrepareModuleInput(
                    input_layouts=Replicate(), 
                    desired_input_layouts=Replicate(),
                ),
                "final_layer.linear": ColwiseParallel(
                    output_layouts=Replicate()
                ),
            }
        )
        # model.fps_embedder.outdim = model.fps_embedder.outdim // tp_mesh.size()
        
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    # TODO: remove this check once PyTorch 2.5 is released. We can safely assume
    # that users won't use a nightly build which is older than 20240809 by then.
    # check_strided_sharding_enabled()

    for layer_id, st_blocks in enumerate(model.st_blocks):
        # As an optimization, do not reshard after forward for the last
        # transformer block since FSDP would prefetch it immediately
        reshard_after_forward = int(layer_id) < len(model.st_blocks) - 1
        fully_shard(
            st_blocks,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config, reshard_after_forward=True)
    return model

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
    
    # understand world topology
    tp_size = int(cfg.get('tp_size', 1))
    _rank = int(os.environ["RANK"])
    _world_size = int(os.environ["WORLD_SIZE"])

    print(f"Starting PyTorch 2D (FSDP + TP) on rank {_rank}.")
    assert (
        _world_size % tp_size == 0
    ), f"World size {_world_size} needs to be divisible by TP size {tp_size}"
    
    # create a sharding plan based on the given world_size.
    dp_size = _world_size // tp_size
    if tp_size > 1:
        mesh_size = (dp_size, tp_size)
        mesh_dims = ("dp", "tp")
    else:
        mesh_size = (dp_size,)
        mesh_dims = ("dp",)
    # Create a device mesh with 2 dimensions.
    # First dim is the data parallel dimension
    # Second dim is the tensor parallel dimension.
    device_mesh = init_device_mesh("cuda", mesh_size, mesh_dim_names=mesh_dims)
    device = torch.cuda.current_device()

    tp_mesh = device_mesh["tp"] if tp_size > 1 else None
    dp_mesh = device_mesh["dp"]

    # For TP, input needs to be same across all TP ranks.
    # while for SP, input can be different across all ranks.
    dp_rank = dp_mesh.get_local_rank()

    set_seed(cfg.get("seed", 1024))

    # == init exp_dir ==
    exp_name, exp_dir = define_experiment_workspace(cfg)
    dist.barrier()
    if dist.get_rank() == 0: # only master should do makedirs
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    dist.barrier()

    # == init logger, tensorboard & wandb ==
    logger = create_logger(exp_dir)
    logger.info(f"Device Mesh created: {device_mesh=}")
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
        process_group=dp_mesh.get_group(),
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
        .to(device) # model is suggested to keep fp32 dtype with AMP and FSDP. Due to: https://github.com/huggingface/accelerate/issues/2624
    )
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
    
    # =======================================================
    # 4. distributed training preparation 
    # =======================================================
    logger.info("Preparing for distributed training...")
    torch.set_default_dtype(dtype)
    mode = cfg.get('mode', 'SP+FSDP')
    local_rank = dist.get_rank() % torch.cuda.device_count()
    if mode == 'DDP':
        model = DDP(model, 
                device_ids             = [local_rank],
                output_device          = local_rank,
                find_unused_parameters = True)
    elif mode == 'SP+FSDP':
        # apply TP+FSDP
        model = parallelize_model(model, tp_mesh, dp_mesh, dtype)
        ema = parallelize_model(ema, tp_mesh, dp_mesh, dtype)

        logger.info(f"Model after parallelization {model=}\n")
    else:
        print("Other training mode besides DDP and FSDP is not supported now.")
        sys.exit(0)
    model.train()
        
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

    # == additional preparation ==    
    if cfg.get("temporal_mask_ratios", None) is not None:
        xt_mask_generator = TemporalMaskGenerator(cfg.temporal_mask_ratios)
    if cfg.get("spatial_mask_ratio", None) is not None:
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)
        xs_mask_generator = SpatialMaskGenerator(cfg.spatial_mask_ratio, patch_size)
            
    # reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    start_epoch = start_step = log_step = acc_step = 0
    running_loss = 0.0
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        ret = shared_load(
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
        "encode",
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
        ) as pbar, implicit_replication():
            for step, batch in pbar:
                timer_list = []
                with timers["move_data"] as move_data_t:
                    x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]
                    y = batch.pop("text")
                if record_time:
                    timer_list.append(move_data_t)

                # == visual and text encoding ==
                with timers["encode"] as encode_t:
                    with torch.no_grad():
                        # Prepare visual inputs
                        if cfg.get("load_video_features", False):
                            x = x.to(device, dtype)
                        else:
                            x = vae.encode(x)  # [B, C, T, H/P, W/P]
                        # Prepare text inputs
                        if cfg.get("load_text_features", False):
                            model_args = {"y": y.to(device, dtype)}
                            mask = batch.pop("mask")
                            if isinstance(mask, torch.Tensor):
                                mask = mask.to(device, dtype)
                            model_args["mask"] = mask
                        else:
                            model_args = text_encoder.encode(y)
                if record_time:
                    timer_list.append(encode_t)

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
                        for n, params in model.named_parameters():
                            torch.nn.utils.clip_grad_norm_(params, cfg.get('grad_clip'), foreach=False)

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
                    log_step = 0

                # == checkpoint saving ==
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    save_dir = sharded_save(
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