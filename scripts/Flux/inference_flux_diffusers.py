import os
import time
from pprint import pformat

import torch
import torch.distributed as dist
from tqdm import tqdm

from vidgen.registry import MODELS, SCHEDULERS, build_module
from vidgen.utils.config_utils import parse_configs

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers import FluxPipeline

def main():
    cfg = parse_configs(training=False)
    dtype = torch.bfloat16
    device="cuda"
    model = (
        build_module(
            cfg.model,
            MODELS,
        )
        .to(device, dtype)
        .eval()
    )
    
    # vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    
    pipeline = FluxPipeline.from_pretrained("/home/dufei.df/workgroup_shanghai/dufei.df/models/aigc/FLUX.1-schnell-diffusers/", torch_dtype=torch.bfloat16)
    
    # import pdb
    # pdb.set_trace()
    pipeline.transformer = model
    pipeline.to(device, dtype=torch.bfloat16)
    
    prompt = "A cat holding a sign that says hello world"
    image = pipeline(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=torch.Generator("cuda").manual_seed(0)
    ).images[0]

    image.save("test.png")


    
    # == build scheduler ==
    # scheduler = build_module(cfg.scheduler, SCHEDULERS)


if __name__ == "__main__":
    main()
