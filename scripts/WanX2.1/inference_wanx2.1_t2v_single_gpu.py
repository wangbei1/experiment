import os
import time
import numpy as np
from pprint import pformat
import functools
from functools import partial
import torch
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
from vidgen.acceleration.parallel_states import set_sequence_parallel_group
from vidgen.datasets import save_sample
from vidgen.datasets.aspect import get_image_size, get_num_frames
from vidgen.models.text_encoder.t5 import text_preprocessing
from vidgen.registry import MODELS, SCHEDULERS, build_module
from vidgen.utils.config_utils import parse_configs
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
from vidgen.utils.ckpt_utils import sharded_load, load_checkpoint
from vidgen.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype
from vidgen.utils.constants import PROMPT_TEMPLATE
from vidgen.models.text_encoder import WanX21T5Encoder


def encode_prompt(
    prompt,
    neg_prompt,
    text_encoder,
    max_seq_len):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    neg_prompt = [neg_prompt] if isinstance(neg_prompt, str) else neg_prompt

    context = text_encoder(prompt)
    context_null = text_encoder(neg_prompt)

    return dict(context=context, context_null=context_null, max_seq_len=max_seq_len)

def save_video_clip(video_clip,
                    prompt,
                    fps,
                    frames,
                    save_path,
                    watermark_flag=False,
                    verbose=True):
    if verbose:
        print(f"Prompt: {prompt}")

    # 1. process video clip
    video_clip = video_clip[:, dframe_to_frame(frames) :]
    video = torch.cat(video_clip, dim=1)
    
    # 2. save videos
    save_path = save_sample(
        video,
        fps=fps,
        save_path=save_path,
        verbose=verbose)

    # 3. add watermark
    if save_path.endswith(".mp4") and watermark_flag:
        time.sleep(1)  # prevent loading previous generated video
        add_watermark(save_path)


def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # parse configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    # dist.init_process_group(backend="nccl")
    
    device_mesh = init_device_mesh("cuda", (1, ))
    
    torch.cuda.set_device(dist.get_rank() % 1)
    set_random_seed(seed=cfg.get("seed", 1024))
    
    device = torch.cuda.current_device()
    num_processes = dist.get_world_size()
    global_rank = dist.get_rank()
    is_main_process_bool = global_rank == 0

    # == init logger ==
    # exp_dir = os.path.join(cfg.model.from_pretrained, "eval")
    # os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(None)
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", True)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==    
    text_encoder = WanX21T5Encoder(
            name=cfg.t5.name,
            text_len=cfg.t5.text_len,
            dtype=cfg.t5.dtype,
            device=device,
            checkpoint_path=cfg.t5.checkpoint_path,
            tokenizer_path=cfg.t5.tokenizer_path)
    
    vae = build_module(cfg.get("vae", None), MODELS)
    vae.model = vae.model.to(device=device, dtype=dtype).eval()

    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    print(f"input_size: {input_size}")
    # vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    # vae_scale_factor_temporal = vae.config.temporal_compression_ratio
        
    # latent_size = [(input_size[0] - 1) // vae_scale_factor_temporal + 1, int(input_size[1]) // vae_scale_factor_spatial, int(input_size[2]) // vae_scale_factor_spatial]
    
    # latent_size = vae.get_latent_size(input_size)

    latent_size = [(input_size[0] - 1) // vae.model.temporal_scale_factor + 1,
                   int(input_size[1]) // vae.model.spatial_scale_factor,
                   int(input_size[2]) // vae.model.spatial_scale_factor]
    
    # model is suggested to keep fp32 dtype with AMP and FSDP. Due to: https://github.com/huggingface/accelerate/issues/2624
    model = (build_module(cfg.model, MODELS).to(dtype).eval())
    if cfg.model.get("from_pretrained", None) is not None:
        load_checkpoint(model, cfg.model.from_pretrained)
    logger.info("Finish create wanx core model")

    mode = cfg.get('mode', 'FSDP')
    local_rank = dist.get_rank() % torch.cuda.device_count()
    if mode == 'DDP':
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    elif mode == 'FSDP':
        fpSixteen = MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float, buffer_dtype=dtype)

        my_auto_wrap_policy = functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in (list(model.blocks)),
        )
        
        model = FSDP(model, mixed_precision=fpSixteen, auto_wrap_policy=my_auto_wrap_policy, device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True, device_mesh=device_mesh)
    
    model.eval()
                
    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    logger.info("Finish create scheduler")
    # ======================================================
    # inference
    # ======================================================
    # == load prompts ==
    prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)
    if prompts is None:
        if cfg.get("prompt_path", None) is not None:
            prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
        else:
            prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop

    prompts = prompts + (prompts[:(num_processes - len(prompts) % num_processes)] if len(prompts) % num_processes != 0 else [])
    # == prepare reference ==
    reference_path = cfg.get("reference_path", [""] * len(prompts))
    mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
    assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
    assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

    # == prepare arguments ==
    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    multi_resolution = cfg.get("multi_resolution", None)
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)
    loop = cfg.get("loop", 1)
    condition_frame_length = cfg.get("condition_frame_length", 5)
    condition_frame_edit = cfg.get("condition_frame_edit", 0.0)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)

    prompts = prompts[global_rank::num_processes]
    mask_strategy = mask_strategy[global_rank::num_processes]
    reference_path = reference_path[global_rank::num_processes]
    # == Iter over all samples ==
    progress_bar = tqdm(range(len(prompts)), disable=not is_main_process_bool)
    progress_bar.set_description("Steps")
    
    logger.info("Begin to process prompts")

    for i in range(0, len(prompts), batch_size):
        # == prepare batch prompts ==
        batch_prompts = prompts[i : i + batch_size]
        ms = mask_strategy[i : i + batch_size]
        refs = reference_path[i : i + batch_size]

        if ";" in batch_prompts[0]:
            batch_prompts_en = []
            for i in range(len(batch_prompts)):
                batch_prompts_en.append(batch_prompts[i].split(";")[0])
                batch_prompts[i] = ";".join(batch_prompts[i].split(";")[1:])
        else:
            batch_prompts_en = batch_prompts
                
        # == get json from prompts ==
        batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
        original_batch_prompts = batch_prompts_en

        # == get reference for condition ==
        refs = collect_references_batch(refs, vae, image_size)

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
        )
        # == Iter over number of sampling for one prompt ==
        for k in range(num_sample):
            # == prepare save paths ==
            save_paths = [
                get_save_path_name(
                    save_dir,
                    sample_name=sample_name,
                    sample_idx=start_idx + idx,
                    prompt=original_batch_prompts[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                    k=k,
                )
                for idx in range(len(batch_prompts))
            ]

            # NOTE: Skip if the sample already exists
            # This is useful for resuming sampling VBench
            if prompt_as_path and all_exists(save_paths):
                continue

            # == process prompts step by step ==
            # 0. split prompt
            # each element in the list is [prompt_segment_list, loop_idx_list]
            batched_prompt_segment_list = []
            batched_loop_idx_list = []
            for prompt in batch_prompts:
                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                batched_prompt_segment_list.append(prompt_segment_list)
                batched_loop_idx_list.append(loop_idx_list)

            # 2. append score
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=cfg.get("aes", None),
                    flow=cfg.get("flow", None),
                    camera_motion=cfg.get("camera_motion", None),
                )

            # 3. clean prompt with T5
            # for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
            #     batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

            # 4. merge to obtain the final prompt
            batch_prompts = []
            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

            # == Iter over loop generation ==
            video_clips = []
            for loop_i in range(loop):
                # == get prompt for loop i ==
                batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

                # == add condition frames for loop ==
                if loop_i > 0:
                    refs, ms = append_generated(
                        vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                    )

                # == sampling ==
                # torch.manual_seed(1024)
                generator = torch.Generator(device=device).manual_seed(cfg.get("seed", 1024))
                # z = torch.randn(len(batch_prompts), vae.z_dim, *latent_size, device=device, generator=generator, dtype=dtype)
                
                z = torch.randn(len(batch_prompts), vae.model.z_dim, *latent_size, device=device, generator=generator, dtype=dtype)

                # masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                
                y = encode_prompt(prompt=batch_prompts_loop,
                                  neg_prompt=cfg.sample_neg_prompt,
                                  text_encoder=text_encoder,
                                  max_seq_len=cfg.max_seq_len)

                samples = scheduler.sample(
                    model,
                    y,
                    z=z,
                    prompts=batch_prompts_loop,
                    device=device,
                    additional_args=model_args,
                    progress=verbose,
                    mask=None,
                    generator=generator,
                    cfg=cfg,
                    mode="t2v"
                )
                samples = vae.decode(samples)
                video_clips.append(samples)

            # == save samples ==
            for idx, batch_prompt in enumerate(batch_prompts):
                if verbose:
                    logger.info("Prompt: %s", batch_prompt)
                save_path = save_paths[idx]
                video = [video_clips[j][idx] for j in range(loop)]
                for j in range(1, loop):
                    video[j] = video[j][:, dframe_to_frame(condition_frame_length) :]
                video = torch.cat(video, dim=1)
                save_path = save_sample(
                    video,
                    fps=save_fps,
                    save_path=save_path,
                    verbose=verbose,
                )
                if save_path.endswith(".mp4") and cfg.get("watermark", False):
                    time.sleep(1)  # prevent loading previous generated video
                    add_watermark(save_path)
        start_idx += len(batch_prompts)
        progress_bar.update(1)
    
    logger.info("Inference finished.")
    logger.info(f"Saved {start_idx} samples to {save_dir}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
