from pprint import pformat
import os
import re
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from mmengine.runner import set_random_seed
from tqdm import tqdm

from vidgen.acceleration.parallel_states import get_data_parallel_group, set_data_parallel_group
from vidgen.datasets.dataloader import prepare_dataloader
from vidgen.registry import DATASETS, MODELS, SCHEDULERS, build_module
from vidgen.utils.config_utils import parse_configs
from vidgen.utils.misc import create_logger, to_torch_dtype
from vidgen.utils.ckpt_utils import load
from vidgen.utils.misc import create_tensorboard_writer
import os
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
import time
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # == init distributed env ==
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_random_seed(seed=cfg.get("seed", 1024))
    set_data_parallel_group(dist.group.WORLD)
    device = torch.cuda.current_device()

    # == init logger ==
    logger = create_logger()
    logger.info("Validation loss configuration:\n %s", pformat(cfg.to_dict()))

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder and vae ==
    text_encoder = CLIPTextModel.from_pretrained(cfg.text_encoder.from_pretrained, subfolder=cfg.text_encoder.subfolder).to(device=device, dtype=dtype).eval()
    text_encoder_2 = T5EncoderModel.from_pretrained(cfg.text_encoder_2.from_pretrained, subfolder=cfg.text_encoder_2.subfolder).to(device=device, dtype=dtype).eval()
    tokenizer = CLIPTokenizer.from_pretrained(cfg.tokenizer.from_pretrained, subfolder=cfg.tokenizer.subfolder)
    tokenizer_2 = T5TokenizerFast.from_pretrained(cfg.tokenizer_2.from_pretrained, subfolder=cfg.tokenizer_2.subfolder)
    # == build vae ==
    vae = build_module(cfg.get("vae", None), MODELS)
    vae = vae.to(device=device, dtype=dtype).eval()

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
    # torch.set_default_dtype(dtype)
    # model = model.to(torch.cuda.current_device())
    mode = cfg.get('mode', 'DDP')
    local_rank = dist.get_rank() % torch.cuda.device_count()
    if mode == 'DDP':
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    elif mode == 'FSDP':
        fpSixteen = MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float, buffer_dtype=dtype)
        my_size_based_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=5e7) 
        model = FSDP(model, mixed_precision=fpSixteen, auto_wrap_policy=my_size_based_auto_wrap_policy, device_id=torch.cuda.current_device(),
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
    
    while True:
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
            if ckpt in finished_ckpts:
                continue
            try:
                load(os.path.join(cfg.exp_dir, ckpt), model)
            except Exception as err:
                print(err)
                continue

            global_step = extract_number(ckpt)
            evaluation_losses = {}
            bucket_config = cfg.bucket_config
            val_loss = 0.
            val_num = 0
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
                    
                    bucket_loss = torch.tensor(0., device=device)
                    num_samples = torch.tensor(0, device=device)
                    # dataloader_iter = iter(dataloader)
                    # for _ in tqdm(range(num_steps_per_epoch), desc=f"res: {res}, num_frames: {num_frames}"):
                    #     batch = next(dataloader_iter)
                    for batch in dataloader:
                        x = batch.pop("video").to(device, dtype)
                        y = batch.pop("text")
                        with torch.no_grad():
                            if x.shape[2] == 1:
                                h, w = x.shape[-2], x.shape[-1]
                                small_batch_size = max(int(36864 / (h * w) * 100), 1)
                                x = torch.cat([vae.encode(x[i:i+small_batch_size, :]).latent_dist.sample() for i in range(0, x.shape[0], small_batch_size)])
                                # x = x.unsqueeze(2)
                            else:
                                total_res = []
                                h, w = x.shape[-2], x.shape[-1]
                                small_batch_size = max(1, int(36864 / (h * w) * 50)) #以144p为基础，可以同时bs=50，但加速效果也不太明显
                                for j in range(0, x.shape[0], small_batch_size):
                                    vae_res = [vae.encode(x[j:j+small_batch_size, :, :9]).latent_dist.sample()]
                                    # res.extend([vae.encode(x[j:j+small_batch_size, :, i:i+8]).latent_dist.sample() for i in range(9, x.shape[2], 8)])
                                    for i in range(9, x.shape[2], 8):
                                        vae_res.append(vae.encode(x[j:j+small_batch_size, :, i:i+8]).latent_dist.sample())
                                    vae_res = torch.cat(vae_res, dim=2)
                                    total_res.append(vae_res)
                                    vae.clear_fake_context_parallel_cache()
                                x = torch.cat(total_res)
                            x = x * vae.config.scaling_factor
                            
                            model_args = encode_prompt(y,
                                    text_encoder,
                                    tokenizer,
                                    text_encoder_2,
                                    tokenizer_2,
                                    tokenizer_max_length=77)
                            # == video meta info ==
                            for k, v in batch.items():
                                model_args[k] = v.to(device, dtype)
                            
                            t = 0.
                            for t in torch.linspace(0, scheduler.num_timesteps, cfg.get("num_eval_timesteps", 10) + 2)[1:-1]:
                                # == diffusion loss computation ==
                                timestep = torch.tensor([t] * x.shape[0], device=device, dtype=dtype)
                                # with torch.autocast(device_type="cuda"):
                                loss_dict = scheduler.training_losses(model, x, model_args, t=timestep)
                                losses = loss_dict["loss"].detach()  # (batch_size)
                                num_samples += x.shape[0]
                                bucket_loss += losses.sum()
                                # print(t, " ", losses.sum(), " ", bucket_loss)
                                # print("ttttt: ", t, losses)
                    
                    dist.reduce(tensor=bucket_loss, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(tensor=num_samples, dst=0, op=dist.ReduceOp.SUM)
                        
                    if dist.get_rank() == 0:
                        print(bucket_loss, " ", num_samples)
                        bucket_loss_avg = bucket_loss / num_samples if torch.sum(num_samples) else torch.tensor(0., device=device)
                        logger.info("Global step: %s, validation losses for resolution: %s, num_frames: %s, loss: %s\n",
                            global_step, res, num_frames, bucket_loss_avg)
                        # tb_writer.add_scalar(f"val_loss/{res}/{num_frames}", bucket_loss_avg, global_step)
                        # update total loss and number
                        val_loss += bucket_loss
                        val_num += num_samples            
            
            # log final val loss
            if dist.get_rank() == 0:
                val_loss_avg = val_loss/val_num
                logger.info(f"Validation loss at {global_step} steps: {val_loss_avg}")
                tb_writer.add_scalar(f"val_loss/total", val_loss_avg, global_step)

                # save ckpt name to process_file
                with open(process_file, 'a') as f:
                    f.write(ckpt+'\n')
                
        time.sleep(300)

            

if __name__ == "__main__":
    main()
