from pprint import pformat
import time
import os
import re
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, lambda_auto_wrap_policy
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
from transformers import (
    AltCLIPTextModel,
    XLMRobertaTokenizer,
    MT5EncoderModel,
    T5TokenizerFast,
    T5EncoderModel
)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def encode_prompt(
    prompt,
    text_encoder,
    tokenizer,
    tokenizer_max_length=256
):
    device = text_encoder.device
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    
    text_input = tokenizer(
        prompt,
        max_length=tokenizer_max_length,  # Max token length for T5 is set here.
        padding="max_length",
        return_tensors="pt",
        truncation=True,
        return_attention_mask=True,
    )
    caption_input_ids_t5 = text_input["input_ids"].to(device)
    caption_attention_mask_t5 = text_input["attention_mask"].to(device)
    
    for i in range(len(prompt)):
        if prompt[i] == "":
            caption_input_ids_t5[i] = 0
            caption_attention_mask_t5[i] = 0
            
    y_mask = [caption_attention_mask_t5.bool()]
    
    y_feat = [text_encoder(caption_input_ids_t5, caption_attention_mask_t5).last_hidden_state]
    
    return {"y_mask": y_mask, "y_feat": y_feat}

def decode_latents(vae, latents, num_frames):
    # latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
    
    latents = 1 / vae.config.scaling_factor * latents + vae.config.shift_factor
    
    frames = vae.decode(latents, num_frames, time_slice=8, batch_slice_2d=1).sample
    
    return frames

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
    if 'mt5' in cfg.text_encoder.from_pretrained:
        logger.info("use mt5 as text encoder.")
        text_encoder = MT5EncoderModel.from_pretrained(cfg.text_encoder.from_pretrained, subfolder=cfg.text_encoder.subfolder).to(device=device, dtype=dtype).eval()
    else:
        logger.info("use t5 as text encoder.")
        text_encoder = T5EncoderModel.from_pretrained(cfg.text_encoder.from_pretrained, subfolder=cfg.text_encoder.subfolder).to(device=device, dtype=dtype).eval()
    tokenizer = T5TokenizerFast.from_pretrained(cfg.tokenizer.from_pretrained, subfolder=cfg.tokenizer.subfolder)
    
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
            lambda_fn=lambda m: m in model.blocks,
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
            val_num = 0.
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
                    num_sample = {key: torch.tensor(0., device=device) for key in categories}
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
                                        tokenizer)
                            
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
                                for txt, loss in zip(y, losses):
                                    category = classify_text(txt)
                                    bucket_loss[category] += loss
                                    num_sample[category] += torch.tensor(1., device=device)
                                # print(t, " ", losses.sum(), " ", bucket_loss)
                                # print("ttttt: ", t, losses)
                    
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
                
        time.sleep(300)

            

if __name__ == "__main__":
    main()
