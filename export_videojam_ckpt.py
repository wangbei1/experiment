#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
把 Wan2.1-T2V-1.3B 的 diffusion_pytorch_model.safetensors
扩成 32 通道 VideoJAM 版本并另存为一个 safetensors。

用法示例：

CUDA_VISIBLE_DEVICES=0 python export_videojam_32ch_ckpt.py \
  --base_ckpt /data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
  --out_ckpt  /data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model_videojam_32ch.safetensors
"""

import os
import argparse

import torch
from safetensors.torch import save_file

from vidgen.registry import MODELS, build_module
from vidgen.utils.ckpt_utils import load_checkpoint


def build_base_wanx_13b(base_channels: int = 16):
    """
    按你训练时的 config 构一个 16 通道的 WanX2.1-T2V-1.3B 基础模型（不带 VideoJAM），
    然后 init_weights 一下，方便后面 load_checkpoint 覆盖参数。
    """
    text_len = 512

    model_cfg = dict(
        type="WanX21",          # 和你训练脚本里用的一致
        patch_size=(1, 2, 2),
        text_len=text_len,
        in_dim=base_channels,   # 16
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=4096,
        out_dim=base_channels,  # 16
        num_heads=12,
        num_layers=30,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        use_fixed_seq_len=False,
        sp_degree=1,            # 随便填，导出 ckpt 不跑分布式
    )

    print("[export] build 16-ch WanX21 base model by registry (type='WanX21')")
    model = build_module(model_cfg, MODELS)

    if hasattr(model, "init_weights"):
        print("[export] init random weights for base model")
        model.init_weights()

    return model


def expand_and_save_videojam_ckpt(
    base_ckpt: str,
    out_ckpt: str,
    base_channels: int = 16,
):
    """
    1. 构建 16ch WanX 基础模型
    2. load_checkpoint(base_ckpt)
    3. 调用 model.expand_to_videojam(base_channels)
    4. 把扩充后的 state_dict 存成 safetensors
    """
    assert os.path.exists(base_ckpt), f"base_ckpt not found: {base_ckpt}"

    # 1. 构建基础模型并加载预训练权重
    model = build_base_wanx_13b(base_channels=base_channels)

    print(f"[export] load base ckpt from: {base_ckpt}")
    load_checkpoint(model, base_ckpt)
    print("[export] base ckpt loaded.")

    # 打个 log 看一下当前 in_dim / out_dim
    print(f"[export] before expand: in_dim={model.in_dim}, out_dim={model.out_dim}")

    # 2. 扩成 VideoJAM 32 通道
    if not hasattr(model, "expand_to_videojam"):
        raise RuntimeError(
            "model 没有 expand_to_videojam 方法，请先在 Transformer 里添加前面那版 expand_to_videojam 再运行本脚本。"
        )

    model.expand_to_videojam(base_channels=base_channels)
    print(f"[export] after expand:  in_dim={model.in_dim}, out_dim={model.out_dim}")

    # 3. 移到 CPU 并导出 state_dict
    model.to("cpu")
    state_dict = model.state_dict()

    out_dir = os.path.dirname(out_ckpt)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    print(f"[export] saving expanded 32-ch VideoJAM ckpt to: {out_ckpt}")
    save_file(state_dict, out_ckpt)
    print("[export] done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Expand Wan2.1-T2V-1.3B diffusion_pytorch_model.safetensors to 32-ch VideoJAM checkpoint."
    )

    parser.add_argument(
        "--base_ckpt",
        type=str,
        default="/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        help="原始 16 通道 WanX diffusion_pytorch_model.safetensors 路径",
    )

    parser.add_argument(
        "--out_ckpt",
        type=str,
        default="/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model_videojam_32ch.safetensors",
        help="扩充后 32 通道 VideoJAM ckpt 的输出路径",
    )

    parser.add_argument(
        "--base_channels",
        type=int,
        default=16,
        help="基础 latent 通道数（Wan2.1-T2V-1.3B 为 16）",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 这里只用 CPU 也行，GPU 对导出没有影响
    torch.set_grad_enabled(False)

    expand_and_save_videojam_ckpt(
        base_ckpt=args.base_ckpt,
        out_ckpt=args.out_ckpt,
        base_channels=args.base_channels,
    )


if __name__ == "__main__":
    main()
