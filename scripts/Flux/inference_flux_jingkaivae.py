import os
import time
from pprint import pformat

import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed
from tqdm import tqdm

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
from vidgen.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype

from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

def encode_prompt(
    prompt,
    text_encoder,
    tokenizer,
    text_encoder_2,
    tokenizer_2,
    tokenizer_max_length=77
):
    device = text_encoder.device
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    
    text_input_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    ).input_ids
    pooled_prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False).pooler_output

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

def decode_latents(vae, latents, num_frames):
    # latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
    latents = 1 / vae.config.scaling_factor * latents
    latents = latents + vae.config.shift_factor
        
    num_latents = latents.shape[2]
    # if num_latents == 1:
    #     frames = vae.decode(latents).sample
    # else:
    #     start_frame, end_frame = (0, 3)
    #     frames = [vae.decode(latents[:, :, start_frame:end_frame]).sample]
    #     for i in range(3, num_latents, 2):
    #         start_frame, end_frame = i, i + 2
    #         current_frames = vae.decode(latents[:, :, start_frame:end_frame]).sample
    #         frames.append(current_frames)
    #     frames = torch.cat(frames, dim=2)
    #     frames = vae.decode(latents).sample
    # vae.l()
    frames = vae.decode(latents, num_frames).sample
    return frames
        
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
    if is_distributed():
        enable_sequence_parallelism = dist.get_world_size() > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        enable_sequence_parallelism = False
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = CLIPTextModel.from_pretrained(cfg.text_encoder.from_pretrained, subfolder=cfg.text_encoder.subfolder).to(device=device, dtype=dtype).eval()
    text_encoder_2 = T5EncoderModel.from_pretrained(cfg.text_encoder_2.from_pretrained, subfolder=cfg.text_encoder_2.subfolder).to(device=device, dtype=dtype).eval()
    tokenizer = CLIPTokenizer.from_pretrained(cfg.tokenizer.from_pretrained, subfolder=cfg.tokenizer.subfolder)
    tokenizer_2 = T5TokenizerFast.from_pretrained(cfg.tokenizer_2.from_pretrained, subfolder=cfg.tokenizer_2.subfolder)
    
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
    
    model = (
        build_module(
            cfg.model,
            MODELS,
        )
        .to(device, dtype).eval() # model is suggested to keep fp32 dtype with AMP and FSDP. Due to: https://github.com/huggingface/accelerate/issues/2624
    )
    
    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

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
    model_name = "/".join(cfg.model.from_pretrained.split("/")[-3:-1])
    # save_dir = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)

    # == Iter over all samples ==
    for i in progress_wrap(range(0, len(prompts), batch_size)):
        # == prepare batch prompts ==
        batch_prompts = prompts[i : i + batch_size]
        ms = mask_strategy[i : i + batch_size]
        refs = reference_path[i : i + batch_size]

        # == get json from prompts ==
        batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
        original_batch_prompts = batch_prompts

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

            # 1. refine prompt by openai
            if cfg.get("llm_refine", False):
                # only call openai API when
                # 1. seq parallel is not enabled
                # 2. seq parallel is enabled and the process is rank 0
                if not enable_sequence_parallelism or (enable_sequence_parallelism and is_main_process()):
                    for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                        batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

                # sync the prompt if using seq parallel
                if enable_sequence_parallelism:
                    dist.barrier()
                    prompt_segment_length = [
                        len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
                    ]

                    # flatten the prompt segment list
                    batched_prompt_segment_list = [
                        prompt_segment
                        for prompt_segment_list in batched_prompt_segment_list
                        for prompt_segment in prompt_segment_list
                    ]

                    # create a list of size equal to world size
                    broadcast_obj_list = [batched_prompt_segment_list] * dist.world_size
                    dist.broadcast_object_list(broadcast_obj_list, 0)

                    # recover the prompt list
                    batched_prompt_segment_list = []
                    segment_start_idx = 0
                    all_prompts = broadcast_obj_list[0]
                    for num_segment in prompt_segment_length:
                        batched_prompt_segment_list.append(
                            all_prompts[segment_start_idx : segment_start_idx + num_segment]
                        )
                        segment_start_idx += num_segment

            # 2. append score
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=cfg.get("aes", None),
                    flow=cfg.get("flow", None),
                    camera_motion=cfg.get("camera_motion", None),
                )

            # 3. clean prompt with T5
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

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
                z = torch.randn(len(batch_prompts), vae.latent_embed_dim, *latent_size, device=device, dtype=dtype)
                
                masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                y = encode_prompt(batch_prompts_loop,
                                text_encoder,
                                tokenizer,
                                text_encoder_2,
                                tokenizer_2,
                                tokenizer_max_length=77)
                y_null = encode_prompt(["" for i in range(len(batch_prompts_loop))],
                                text_encoder,
                                tokenizer,
                                text_encoder_2,
                                tokenizer_2,
                                tokenizer_max_length=77)
                
                samples = scheduler.sample(
                    model,
                    y,
                    y_null,
                    z=z,
                    prompts=batch_prompts_loop,
                    device=device,
                    additional_args=model_args,
                    progress=verbose >= 2,
                    mask=masks,
                )
                
                samples = decode_latents(vae, samples.to(dtype), num_frames)
                video_clips.append(samples)

            # == save samples ==
            if is_main_process():
                for idx, batch_prompt in enumerate(batch_prompts):
                    if verbose >= 2:
                        logger.info("Prompt: %s", batch_prompt)
                    save_path = save_paths[idx]
                    save_path = save_path+f"_{num_frames}"
                    video = [video_clips[i][idx] for i in range(loop)]
                    for i in range(1, loop):
                        video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]
                    video = torch.cat(video, dim=1)
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
    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx, save_dir)


if __name__ == "__main__":
    main()
