import os
import time
from pprint import pformat
import functools
from functools import partial
import torch
from datetime import timedelta
import random
import re
import numpy as np
import torch.distributed as dist
from vidgen.utils.train_utils import set_random_seed
from tqdm import tqdm
from typing import Dict, List
import torch.nn.functional as F
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
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, lambda_auto_wrap_policy
from vidgen.acceleration.parallel_states import get_data_parallel_group, set_sequence_parallel_group
from vidgen.datasets import save_sample
from vidgen.datasets.dataloader import prepare_dataloader
from vidgen.datasets.aspect import get_image_size, get_num_frames
from vidgen.models.text_encoder.t5 import text_preprocessing
from vidgen.registry import DATASETS, MODELS, SCHEDULERS, build_module
from vidgen.utils.config_utils import parse_configs
from vidgen.utils.misc import create_tensorboard_writer
from vidgen.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from vidgen.utils.ckpt_utils import load, sharded_load
from vidgen.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype
from vidgen.utils.constants import PROMPT_TEMPLATE
from vidgen.models.text_encoder import HunyuanTextEncoder

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def encode_prompt(
    prompt,
    text_encoder,
    text_encoder_2,
    data_type, #image, video
):
    device = text_encoder.device
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    
    text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)
    prompt_outputs = text_encoder.encode(
        text_inputs, data_type=data_type, device=device
    )
    prompt_embeds = prompt_outputs.hidden_state
    attention_mask = prompt_outputs.attention_mask
    
    text_inputs_2 = text_encoder_2.text2tokens(prompt, data_type=data_type)
    prompt_outputs_2 = text_encoder_2.encode(
        text_inputs_2, data_type=data_type, device=device
    )
    prompt_embeds_2 = prompt_outputs_2.hidden_state
    
    return dict(text_states=prompt_embeds,  # [bs, 256, 4096]
                text_mask=attention_mask,  # [bs, 256]
                text_states_2=prompt_embeds_2)
                
def decode_latents(vae, latents, num_frames):

    latents = 1 / vae.config.scaling_factor * latents
    latents = latents.to(vae.dtype)
    vae.enable_tiling()
    image = vae.decode(latents, return_dict=False)[0]
    
    return image

def build_dataset(cfg, resolution, num_frames, batch_size):
    bucket_config = {resolution: {num_frames: (1.0, batch_size)}}
    dataset = build_module(cfg.dataset, DATASETS)
    dataloader_args = dict(
        dataset=dataset,
        batch_size=None,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
    )
    dataloader, sampler = prepare_dataloader(bucket_config=bucket_config, **dataloader_args)
    num_batch = sampler.get_num_batch()
    num_steps_per_epoch = num_batch // dist.get_world_size()
    return dataloader, num_steps_per_epoch, num_batch

        
def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    assert torch.cuda.is_available(), "Validating currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    # == init distributed training ==
    _rank = int(os.environ["RANK"])
    _world_size = int(os.environ["WORLD_SIZE"])
    
    #初始化一个process_group，可以避免使用torch2.5的device_mesh_flatten
    dist.init_process_group("cpu:gloo,cuda:nccl", timeout=timedelta(hours=24))
    device_num = torch.cuda.device_count()
    mesh_size = (_world_size // device_num, device_num)
    mesh_dims = ("rep", "shard")
    device_mesh = init_device_mesh("cuda", mesh_size, mesh_dim_names=mesh_dims)
    # device_mesh._flatten("dp")
    
    set_seed(cfg.get("seed", 1024))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    device = torch.cuda.current_device()

    # == init logger ==
    logger = create_logger()
    logger.info("Validation loss configuration:\n %s", pformat(cfg.to_dict()))

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = HunyuanTextEncoder(
        text_encoder_type=cfg.text_encoder.text_encoder_type,
        text_encoder_path=cfg.text_encoder.text_encoder_path,
        max_length=cfg.text_encoder.max_length + PROMPT_TEMPLATE['image']['crop_start'],
        max_length_video=cfg.text_encoder.max_length + PROMPT_TEMPLATE['video']['crop_start'],
        text_encoder_precision=cfg_dtype,
        tokenizer_type=cfg.text_encoder.tokenizer_type,
        prompt_template=PROMPT_TEMPLATE['image'],
        prompt_template_video=PROMPT_TEMPLATE['video'],
        hidden_state_skip_layer=cfg.text_encoder.hidden_state_skip_layer,
        logger=logger,
        device=device,
    )
    text_encoder_2 = HunyuanTextEncoder(
        text_encoder_type=cfg.text_encoder_2.text_encoder_type,
        text_encoder_path=cfg.text_encoder_2.text_encoder_path,
        max_length=cfg.text_encoder_2.max_length,
        text_encoder_precision=cfg_dtype,
        tokenizer_type=cfg.text_encoder_2.tokenizer_type,
        logger=logger,
        device=device,
    )
    
    # == build vae ==
    vae = build_module(cfg.get("vae", None), MODELS)
    vae = vae.to(device=device, dtype=dtype).eval()
    # vae.enable_tiling()
    # vae.enable_slicing()

    # == build diffusion model ==
    model = (
        build_module(
            cfg.model,
            MODELS,
        )
        .to(dtype).eval() # model is suggested to keep fp32 dtype with AMP and FSDP. Due to: https://github.com/huggingface/accelerate/issues/2624
    )
    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    # == distributed preparation 
    logger.info("Preparing for distributed validation...")
    
    mode = cfg.get('mode', 'FSDP')
    local_rank = dist.get_rank() % torch.cuda.device_count()
    if mode == 'DDP':
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    elif mode == 'FSDP':
        fpSixteen = MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float, buffer_dtype=dtype)
        # my_size_based_auto_wrap_policy = functools.partial(
        #     size_based_auto_wrap_policy, min_num_params=1e7) 
        my_auto_wrap_policy = functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in (list(model.double_blocks) + list(model.single_blocks)),
        )
        model = FSDP(model, mixed_precision=fpSixteen, auto_wrap_policy=my_auto_wrap_policy, device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True)
    else:
        print("Other training mode besides DDP and FSDP is not supported now.")
        sys.exit(0)
    
    # ======================================================
    # enumerate all ckpts for vlidatation loss 
    # ======================================================
    logger.info(f'Validating {cfg.exp_dir}')
    model.eval()
    if dist.get_rank() == 0:
        tb_writer = create_tensorboard_writer(cfg.exp_dir)
        
    categories = {
        "people": ["man", "woman", "child", "person", "people", "doctor", "teacher", "police"],
        "animals": ["dog", "cat", "elephant", "bird", "fish", "lion", "tiger", "animal", "wildlife", "pet"],
        "landscape": ["mountain", "river", "forest", "beach", "desert", "sky", "ocean", "cityscape"],
        "architecture": ["building", "bridge", "tower", "skyscraper", "house", "temple"],
        "vehicles": ["car", "bike", "airplane", "train", "bus", "ship"],
        "food": ["apple", "pizza", "cake", "meal", "fruit", "vegetables"]
    }
    def classify_text(text):
        text = text.lower()  # 将文本转为小写，方便匹配
        for category, keywords in categories.items():
            for keyword in keywords:
                if re.search(r'\b' + keyword + r'\b', text):  # 使用正则表达式确保匹配完整词
                    return category
        return "unknown"

    process_file = os.path.join(cfg.exp_dir, "processed_ckpts.txt")
    if os.path.exists(process_file):
        with open(process_file, 'r') as f:
            finished_ckpts = f.readlines()
        finished_ckpts = set([file.strip() for file in finished_ckpts])
    else:
        finished_ckpts = []
    ckpt_paths = os.listdir(cfg.exp_dir)
    ckpt_paths = [f for f in ckpt_paths if f.startswith("epoch") ]
    def extract_number(s):
        match = re.search(r'global_step(\d+)', s)
        return int(match.group(1)) if match else 0
    ckpt_paths = sorted(ckpt_paths, key=extract_number)        

    for ckpt in ckpt_paths:
        if not os.path.exists(os.path.join(cfg.exp_dir, ckpt, 'model', '.metadata')):
            continue
        if ckpt in finished_ckpts:
            continue
        try:
            sharded_load(os.path.join(cfg.exp_dir, ckpt), ema=model)
        except Exception as err:
            print(err)
            continue

        global_step = extract_number(ckpt)
        evaluation_losses = {}
        bucket_config = cfg.bucket_config
        val_loss = 0.
        val_num = 0.
        step_losses = {key: torch.tensor(0., device=device) for key in range(cfg.get("num_eval_timesteps", 10))}
        total_num_samples = torch.tensor(0, device=device)
        bucket_losses = {key: torch.tensor(0., device=device) for key in categories}
        num_samples = {key: torch.tensor(0., device=device) for key in categories}
        for i, res in enumerate(bucket_config):
            t_bucket = bucket_config[res]
            for num_frames, (_, batch_size) in t_bucket.items():
                if batch_size is None:
                    continue
                logger.info("Evaluating resolution: %s, num_frames: %s", res, num_frames)
                dataloader, num_steps_per_epoch, num_batch = build_dataset(cfg, res, num_frames, batch_size)
                if num_batch == 0:
                    logger.warning("No data for resolution: %s, num_frames: %s", res, num_frames)
                    continue
                
                bucket_loss = {key: torch.tensor(0., device=device) for key in categories}
                step_loss = {key: torch.tensor(0., device=device) for key in range(cfg.get("num_eval_timesteps", 10))}
                num_sample = {key: torch.tensor(0., device=device) for key in categories}
                total_num_sample = torch.tensor(0, device=device)
                dataloader_iter = iter(dataloader)
                for _ in tqdm(range(num_steps_per_epoch), desc=f"res: {res}, num_frames: {num_frames}"):
                    batch = next(dataloader_iter)
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
                                    text_encoder_2,
                                    data_type="image" if x.shape[2] == 1 else "video")
                        
                        # == video meta info ==
                        for k, v in batch.items():
                            model_args[k] = v.to(device, dtype)
                        
                        t = 0.
                        for xid, t in enumerate(torch.linspace(0, scheduler.num_timesteps, cfg.get("num_eval_timesteps", 10) + 2)[1:-1]):
                            # == diffusion loss computation ==
                            timestep = torch.tensor([t] * x.shape[0], device=device, dtype=dtype)
                            # with torch.autocast(device_type="cuda"):
                            loss_dict = scheduler.training_losses(model, x, model_args, t=timestep)
                            losses = loss_dict["loss"].detach()  # (batch_size)
                            for txt, loss in zip(y, losses):
                                category = classify_text(txt)
                                bucket_loss[category] += loss
                                num_sample[category] += torch.tensor(1., device=device)
                            
                            step_loss[xid] += losses.sum()
                            # print(t, " ", losses.sum(), " ", bucket_loss)
                            # print("ttttt: ", t, losses)
                        total_num_sample += x.shape[0]
                        
                for xid in step_loss.keys():
                    dist.reduce(tensor=step_loss[xid], dst=0, op=dist.ReduceOp.SUM)
                    if dist.get_rank() == 0:
                        step_losses[xid] += step_loss[xid]
                dist.reduce(tensor=total_num_sample, dst=0, op=dist.ReduceOp.SUM)
                if dist.get_rank() == 0:
                    total_num_samples += total_num_sample
                    
                for cat in bucket_loss.keys():
                    dist.reduce(tensor=bucket_loss[cat], dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(tensor=num_sample[cat], dst=0, op=dist.ReduceOp.SUM)
                    
                    num_samples[cat] += num_sample[cat]
                    bucket_losses[cat] += bucket_loss[cat]
                    if dist.get_rank() == 0:
                        val_cat = bucket_loss[cat]/num_sample[cat]
                        logger.info(f"Global step: {global_step}, resolution: {res}, num_frames: {num_frames}, validation losses for {cat}: {val_cat}")
        
        # log final val loss
        if dist.get_rank() == 0:
            for xid, x_loss in step_losses.items():
                val_step_loss = x_loss / total_num_samples
                tb_writer.add_scalar(f"val_loss/step_{xid}", val_step_loss, global_step)
            
            for cat, cat_loss in bucket_losses.items():
                cat_num = num_samples[cat]
                val_cat = cat_loss/cat_num
                tb_writer.add_scalar(f"val_loss/{cat}", val_cat, global_step)
                
                val_loss += cat_loss
                val_num += cat_num
                
            val_loss_avg = val_loss/val_num
            logger.info(f"Global step: {global_step}, validation loss for all category: {val_loss_avg}")
            tb_writer.add_scalar(f"val_loss/total", val_loss_avg, global_step)

            
            # save ckpt name to process_file
            with open(process_file, 'a') as f:
                f.write(ckpt+'\n')
                
            

if __name__ == "__main__":
    main()
