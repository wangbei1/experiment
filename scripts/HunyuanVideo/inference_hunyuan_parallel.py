import os
import time
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
from vidgen.utils.ckpt_utils import load, sharded_load
from vidgen.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype
from vidgen.utils.constants import PROMPT_TEMPLATE
from vidgen.models.text_encoder import HunyuanTextEncoder

from training_acc.parallelisms import parallelize
from training_acc.config import ParallelConfig
from training_acc.dist import initialize, parallel_state, log_rank
from training_acc.logger import logger as acc_logger

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

def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
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
    
    device_mesh = init_device_mesh("cuda", (torch.cuda.device_count(), ))
    
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_random_seed(seed=cfg.get("seed", 1024))
    
    device = torch.cuda.current_device()
    is_main_process_bool = dist.get_rank() == 0
    
    # parallel init
    sp_degree = cfg.get("sp_degree", 1)
    parallel_config = ParallelConfig(sp_degree = sp_degree)
    initialize(parallel_config=parallel_config)
    
    data_parallel_size = parallel_state.get_data_parallel_size()
    data_parallel_rank = parallel_state.get_data_parallel_rank()
    sp_parallel_rank = parallel_state.get_sequence_parallel_rank()

    # == init logger ==
    # exp_dir = os.path.join(cfg.model.from_pretrained, "eval")
    # os.makedirs(exp_dir, exist_ok=True)
    logger = create_logger(None)
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

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
    
    vae = build_module(cfg.get("vae", None), MODELS)
    vae = vae.to(device=device, dtype=dtype).eval()

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
    # vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    # vae_scale_factor_temporal = vae.config.temporal_compression_ratio
        
    # latent_size = [(input_size[0] - 1) // vae_scale_factor_temporal + 1, int(input_size[1]) // vae_scale_factor_spatial, int(input_size[2]) // vae_scale_factor_spatial]
    
    latent_size = vae.get_latent_size(input_size)
    
    ckpt_path = cfg.model.from_pretrained #这里不需要再加ema了
    # cfg.model.from_pretrained = None
    model = (
        build_module(
            cfg.model,
            MODELS,
        )
        .to(dtype).eval() # model is suggested to keep fp32 dtype with AMP and FSDP. Due to: https://github.com/huggingface/accelerate/issues/2624
    )
    
    model = parallelize("hunyuan", model)
    
    mode = cfg.get('mode', 'FSDP')
    local_rank = dist.get_rank() % torch.cuda.device_count()
    if mode == 'DDP':
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    elif mode == 'FSDP':
        fpSixteen = MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float, buffer_dtype=dtype)

        my_auto_wrap_policy = functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in (list(model.double_blocks) + list(model.single_blocks)),
        )
        
        model = FSDP(model, mixed_precision=fpSixteen, auto_wrap_policy=my_auto_wrap_policy, device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True, device_mesh=device_mesh)
    model.eval()
    
    # sharded_load(ckpt_path, ema=model)
                
    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ======================================================
    # inference
    # ======================================================
    s_t = time.time()
    # == load prompts ==
    prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)
    if prompts is None:
        if cfg.get("prompt_path", None) is not None:
            prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
        else:
            prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop

    prompts = prompts + (prompts[:(data_parallel_size - len(prompts) % data_parallel_size)] if len(prompts) % data_parallel_size != 0 else [])
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
    align = cfg.get("align", None)

    save_dir = cfg.save_dir
    # model_name = "/".join(cfg.model.from_pretrained.split("/")[-3:-1])
    # save_dir = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)

    prompts = prompts[data_parallel_rank::data_parallel_size]
    mask_strategy = mask_strategy[data_parallel_rank::data_parallel_size]
    reference_path = reference_path[data_parallel_rank::data_parallel_size]
    # == Iter over all samples ==
    progress_bar = tqdm(range(len(prompts)), disable=not is_main_process_bool)
    progress_bar.set_description("Steps")
    
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
        
        if verbose >= 2:
            acc_logger.info(log_rank(f"idx:{i}, batch_prompts: {batch_prompts}"))
                
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
                z = torch.randn(len(batch_prompts), vae.latent_embed_dim, *latent_size, device=device, generator=generator, dtype=dtype)
                
                masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                
                y = encode_prompt(batch_prompts_loop + [""],
                                text_encoder,
                                text_encoder_2,
                                data_type="image" if z.shape[2] == 1 else "video")
                                                
                samples = scheduler.sample(
                    model,
                    y,
                    z=z,
                    prompts=batch_prompts_loop,
                    device=device,
                    additional_args=model_args,
                    progress=verbose >= 2,
                    mask=masks,
                )
                
                samples = decode_latents(vae, samples.to(dtype), num_frames)
                video_clips.append(samples)

            if sp_parallel_rank == 0:
                # == save samples ==
                for idx, batch_prompt in enumerate(batch_prompts):
                    save_path = save_paths[idx]
                    video = [video_clips[j][idx] for j in range(loop)]
                    for j in range(1, loop):
                        video[j] = video[j][:, dframe_to_frame(condition_frame_length) :]
                    video = torch.cat(video, dim=1)
                    fst, sed = os.path.splitext(save_path)
                    save_path = f"{fst}_sp{sp_degree}_{image_size[0]}x{image_size[1]}{sed}"
                    save_path = save_sample(
                        video,
                        fps=save_fps,
                        save_path=save_path,
                        verbose=verbose >= 2,
                    )
                    if save_path.endswith(".mp4") and cfg.get("watermark", False):
                        time.sleep(1)  # prevent loading previous generated video
                        add_watermark(save_path)
        start_idx += len(batch_prompts)
        progress_bar.update(1)
    e_t = time.time()
    
    logger.info(f"Inference finished. Per prompt time:{(e_t-s_t)/len(prompts):.2f}")
    logger.info("Saved %s samples to %s", start_idx, save_dir)


if __name__ == "__main__":
    main()
