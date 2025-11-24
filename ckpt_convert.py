#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.distributed as dist
from safetensors.torch import save_file

from vidgen.registry import MODELS, build_module
from vidgen.utils.ckpt_utils import sharded_load


def init_single_process_dist(backend="nccl"):
    if not dist.is_available():
        raise RuntimeError("torch.distributed 不可用")

    if dist.is_initialized():
        return

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29444")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    dist.init_process_group(backend=backend, rank=0, world_size=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 DCP 分片 checkpoint 导出单文件 safetensors（model / ema）"
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        required=True,
        help="epoch 目录（不要带 all），例如：/path/.../epoch0-global_step300",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="torch.distributed backend，默认 nccl",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="导出时使用的设备，比如 cuda:0",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_root = args.ckpt_root
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(f"ckpt_root 不是有效目录: {ckpt_root}")

    # 取 ckpt_root 最底层目录名，例如 epoch0-global_step300
    basename = os.path.basename(os.path.normpath(ckpt_root))

    # 导出的单文件路径：放在同一目录下
    out_model_path = os.path.join(ckpt_root, f"{basename}_model.safetensors")
    out_ema_path   = os.path.join(ckpt_root, f"{basename}_ema.safetensors")

    print(f"[info] ckpt_root = {ckpt_root}")
    print(f"[info] basename  = {basename}")
    print(f"[info] will save model to: {out_model_path}")
    print(f"[info] will save ema   to: {out_ema_path}")

    init_single_process_dist(args.backend)

    device = torch.device(args.device)
    dtype = torch.bfloat16

    # 3) 构建和训练时一模一样的 VideoJAM 模型
    text_len = 512
    model_cfg = dict(
        type="WanX21",
        patch_size=(1, 2, 2),
        text_len=text_len,
        in_dim=16,
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=12,
        num_layers=30,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        use_fixed_seq_len=False,
        sp_degree=1,
        num_modalities=2,

    )
    model = build_module(model_cfg, MODELS).to(device=device, dtype=dtype)
    ema = build_module(model_cfg, MODELS).to(device=device, dtype=dtype)

    # 4) 用 sharded_load 把 distcp 分片合并还原到单模型里
    print(f"[info] sharded_load from: {ckpt_root}")
    _ = sharded_load(
        ckpt_root,
        model=model,
        ema=ema,
        optimizer=None,
        lr_scheduler=None,
        sampler=None,
    )

    # 5) 导出普通 state_dict
    os.makedirs(os.path.dirname(out_model_path), exist_ok=True)

    model_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    ema_sd   = {k: v.cpu() for k, v in ema.state_dict().items()}

    save_file(model_sd, out_model_path)
    save_file(ema_sd, out_ema_path)

    print(f"[export] model weights saved to: {out_model_path}")
    print(f"[export] ema   weights saved to: {out_ema_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
