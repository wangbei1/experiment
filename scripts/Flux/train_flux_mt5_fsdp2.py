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
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp import StateDictType
# from torch.distributed.fsdp.fully_sharded_data_parallel import (
#     BackwardPrefetch,
#     CPUOffload,
#     MixedPrecision,
#     ShardingStrategy,
# )
# from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
# from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
import wandb
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from vidgen.acceleration.checkpoint import set_grad_checkpoint
from vidgen.acceleration.parallel_states import get_data_parallel_group
from vidgen.datasets.dataloader import prepare_dataloader
from vidgen.registry import DATASETS, MODELS, SCHEDULERS, build_module
from vidgen.utils.ckpt_utils import sharded_load, model_gathering, model_sharding, record_model_param_shape, sharded_save
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
from transformers import (
    AltCLIPTextModel,
    XLMRobertaTokenizer,
    MT5EncoderModel,
    T5TokenizerFast,
    AutoTokenizer
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def encode_prompt(
    prompt,
    text_encoder,
    tokenizer,
    text_encoder_2,
    tokenizer_2,
    tokenizer_max_length=512
):
    device = text_encoder.device
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )
    text_input_ids, attention_mask = text_input.input_ids, text_input.attention_mask
    pooled_prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask.to(device), output_hidden_states=False).pooler_output

    text_input_ids_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    ).input_ids
        
    prompt_embeds = text_encoder_2(text_input_ids_2.to(device), output_hidden_states=False)[0]

    return {"encoder_hidden_states": prompt_embeds, "pooled_projections": pooled_prompt_embeds}

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

    for layer_id, st_blocks in enumerate(model.transformer_blocks):
        # As an optimization, do not reshard after forward for the last
        # transformer block since FSDP would prefetch it immediately
        reshard_after_forward = int(layer_id) < len(model.transformer_blocks) - 1
        fully_shard(
            st_blocks,
            **fsdp_config,
            reshard_after_forward=True,
        )
    for layer_id, st_blocks in enumerate(model.single_transformer_blocks):
        # As an optimization, do not reshard after forward for the last
        # transformer block since FSDP would prefetch it immediately
        reshard_after_forward = int(layer_id) < len(model.single_transformer_blocks) - 1
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
    # NOTE: A very large timeout is set to avoid some processes exit early
    # dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    _rank = int(os.environ["RANK"])
    _world_size = int(os.environ["WORLD_SIZE"])

    tp_degree = int(cfg.get('tp_degree', 1))
    sp_degree = int(cfg.get('sp_degree', 1))
    # TODO: need to support valid number verification
    dp_degree = _world_size // tp_degree // sp_degree
    if tp_degree > 1:
        mesh_size = (dp_degree, tp_degree, sp_degree)
        mesh_dims = ("dp", "tp", "sp")
        print("tp_size > 1 is unsupported now, please use fsdp+tp insteed.")
        sys.exit(0)
    elif sp_degree > 1:
        mesh_size = (dp_degree, sp_degree)
        mesh_dims = ("dp", "sp")
    else:
        mesh_size = (dp_degree,)
        mesh_dims = ("dp", )
    
    device_mesh = init_device_mesh("cuda", mesh_size, mesh_dim_names=mesh_dims)
    tp_mesh = device_mesh["tp"] if tp_degree > 1 else None
    sp_mesh = device_mesh["sp"] if sp_degree > 1 else None
    # dp_mesh should be used for FSDP. FSDP should be done on dp + sp dims. 
    # see https://github.com/microsoft/Megatron-DeepSpeed/blob/main/pretrain_gpt.py#L39
    # and https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L95
    dp_mesh = (
            device_mesh[("dp", "sp")]._flatten("dp_sp")
            if sp_degree > 1
            else device_mesh["dp"]
        )
    # print(dp_mesh)
    
    # set_seed(cfg.get("seed", 1024))
    device = torch.cuda.current_device()
            
    # ulysses sp
    enable_sp = False
    if sp_degree > 1:
        from training_acc.logger import logger as acc_logger
        from training_acc.utils import set_print_precision
        from training_acc.dist import initialize
        from training_acc.config import ParallelConfig
        
        acc_logger.info("enable ulysses sequence parallel.")
        enable_sp = True
        set_print_precision()
        
        parallel_config = ParallelConfig(sp_degree = sp_degree)
        initialize(parallel_config = parallel_config)
        
    monitor = cfg.get("monitor", False)
    
    data_parallel_group = device_mesh["dp"].get_group()
    if enable_sp:
        from training_acc.dist import parallel_state
        data_parallel_group = parallel_state.get_data_parallel_group()
    
    set_seed(cfg.get("seed", 1024) + data_parallel_group.rank())

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
    
    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    
    # == build text-encoder and vae ==
    text_encoder = AltCLIPTextModel.from_pretrained(cfg.text_encoder.from_pretrained, subfolder=cfg.text_encoder.subfolder).to(device=device, dtype=dtype).eval()
    text_encoder_2 = MT5EncoderModel.from_pretrained(cfg.text_encoder_2.from_pretrained, subfolder=cfg.text_encoder_2.subfolder).to(device=device, dtype=dtype).eval()
    tokenizer = XLMRobertaTokenizer.from_pretrained(cfg.tokenizer.from_pretrained, subfolder=cfg.tokenizer.subfolder)
    tokenizer_2 = T5TokenizerFast.from_pretrained(cfg.tokenizer_2.from_pretrained, subfolder=cfg.tokenizer_2.subfolder)
    # == build vae ==
    vae = build_module(cfg.get("vae", None), MODELS)
    vae = vae.to(device=device, dtype=dtype).eval()

    # == build diffusion model ==
    model = (
        build_module(
            cfg.model,
            MODELS,
        ) # model is suggested to keep fp32 dtype with AMP and FSDP. Due to: https://github.com/huggingface/accelerate/issues/2624
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
    ema = deepcopy(model).to(torch.float32)
    requires_grad(ema, False)
    ema.eval()
    update_ema(ema, model, decay=0)
    
    if enable_sp:
        from training_acc.parallel_strategy import parallelize_flux_transformer
        model = parallelize_flux_transformer(model, parallel_config)
    
    # == setup loss function, build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

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
        model = parallelize_model(model, tp_mesh, dp_mesh, dtype)
        ema = parallelize_model(ema, tp_mesh, dp_mesh, dtype)
    else:
        print("Other training mode besides DDP and FSDP is not supported now.")
        sys.exit(0)
        
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

    prompt_uncond_prob = cfg.get("prompt_uncond_prob", 0.1)
    # == additional preparation ==    
    if cfg.get("temporal_mask_ratios", None) is not None:
        xt_mask_generator = TemporalMaskGenerator(cfg.temporal_mask_ratios)
    if cfg.get("spatial_mask_ratio", None) is not None:
        patch_size = model.config.patch_size
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
    # scaler = ShardedGradScaler() if mode=='FSDP' else GradScaler()
    smoothing_running_loss_avg = None
    torch.cuda.empty_cache()
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
                with timers["encode_vae"] as encode_vae_t:
                    with torch.no_grad():
                        # Prepare visual inputs
                        x_shape_record = x.shape
                        if cfg.get("load_video_features", False):
                            x = x.to(device, dtype)
                        else:
                            if x.shape[2] == 1:
                                h, w = x.shape[-2:]
                                batch_slice_2d = max(2 ** 22 // h // w, 1)
                                x = vae.encode(x, batch_slice_2d=batch_slice_2d).latent_dist.sample()
                            else:
                                h, w = x.shape[-2:]
                                batch_slice_2d = max(2 ** 22 // h // w, 1)
                                x = vae.encode(x, time_slice=32, batch_slice_2d=batch_slice_2d).latent_dist.sample()
                            
                            # x = vae.encode(x)
                        x = (x - vae.config.shift_factor) * vae.config.scaling_factor
                        # x = x * vae.config.scaling_factor
                        # Prepare text inputs
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
                            model_args = encode_prompt(y,
                                text_encoder,
                                tokenizer,
                                text_encoder_2,
                                tokenizer_2)
                            
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
                with timers["diffusion"] as loss_t:
                    loss_dict = scheduler.training_losses(model, x, model_args, xt_mask=xt_mask, xs_mask=xs_mask,
                                                          mae_loss_coef=cfg.get("mae_loss_coef", None), 
                                                          unpatchify_loss=cfg.get("unpatchify_loss", False))
                if record_time:
                    timer_list.append(loss_t)

                # == backward & update ==
                with timers["backward"] as backward_t:
                    loss = loss_dict["loss"].mean()
                    
                    optimizer.zero_grad()
                    
                    loss.backward()
                    
                    if cfg.get('grad_clip', None) is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get('grad_clip'))
                    
                    all_reduce_mean(loss)
                    
                    is_model_update = False
                    if smoothing_running_loss_avg is None or loss.item() < 1.5 * smoothing_running_loss_avg:
                        smoothing_running_loss_avg = (smoothing_running_loss_avg * 0.9 + loss.item() * 0.1) if smoothing_running_loss_avg is not None else loss.item()
                        # scaler.unscale_(optimizer)
                        optimizer.step()
                        
                        is_model_update = True
                        # update learning rate
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                    else:
                        logger.info(f"skip update model because of large loss {loss.item()}, smoothing_running_loss_avg {smoothing_running_loss_avg}")
                                        
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
                        pbar.set_postfix({"loss": avg_loss, "smooth_loss": smoothing_running_loss_avg, "rf_loss": rf_avg_loss, "mae_loss": mae_avg_loss, "lr": optimizer.param_groups[0]["lr"], "step": step, "global_step": global_step})
                        if is_model_update:
                            tb_writer.add_scalar("rf_loss", rf_loss.item(), global_step)
                            tb_writer.add_scalar("mae_loss", mae_loss.item(), global_step)
                    else:
                        # progress bar
                        pbar.set_postfix({"loss": avg_loss, "smooth_loss": smoothing_running_loss_avg, "lr": optimizer.param_groups[0]["lr"], "step": step, "global_step": global_step})
                        
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
                                    if x.shape[2] == 1:
                                        h, w = x.shape[-2:]
                                        batch_slice_2d = max(2 ** 22 // h // w, 1)
                                        x = vae.encode(x, batch_slice_2d=batch_slice_2d).latent_dist.sample()
                                    else:
                                        h, w = x.shape[-2:]
                                        batch_slice_2d = max(2 ** 22 // h // w, 1)
                                        x = vae.encode(x, time_slice=32, batch_slice_2d=batch_slice_2d).latent_dist.sample()
                                    
                                    x = (x - vae.config.shift_factor) * vae.config.scaling_factor
                                    
                                    model_args = encode_prompt(y,
                                            text_encoder,
                                            tokenizer,
                                            text_encoder_2,
                                            tokenizer_2)
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


if __name__ == "__main__":
    main()
