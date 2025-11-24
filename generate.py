#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random

import torch
from torchvision.io import write_video
from mmengine.config import Config

from vidgen.registry import MODELS, SCHEDULERS, build_module
from vidgen.models.text_encoder import WanX21T5Encoder
from vidgen.datasets.aspect import get_image_size
from vidgen.utils.misc import to_torch_dtype
from vidgen.utils.ckpt_utils import load_checkpoint


# =========================
# 工具函数
# =========================

def encode_prompt_inference(
    prompt,
    neg_prompt,
    text_encoder,
    max_seq_len,
):
    """
    和 train_wanx2.1_t2v.py 里的推理版一致：
    返回 dict(context, context_null, max_seq_len)
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    neg_prompt = [neg_prompt] if isinstance(neg_prompt, str) else neg_prompt

    context = text_encoder(prompt)          # 条件分支
    context_null = text_encoder(neg_prompt) # 无条件分支

    return dict(context=context, context_null=context_null, max_seq_len=max_seq_len)


def save_video_tensor_as_mp4(video: torch.Tensor, save_path: str, fps: int = 24):
    """
    video:
      - [B, 3, T, H, W] 或
      - [3, T, H, W]
    数值范围假设在 [-1, 1] 或 [0, 1]
    """
    if video.ndim == 5:  # [B, 3, T, H, W]
        video = video[0]
    assert video.ndim == 4, f"Expect [3,T,H,W], got {video.shape}"

    video = video.detach().cpu()

    # [-1,1] -> [0,1]
    if video.min() < 0.0:
        video = (video.clamp(-1, 1) + 1) / 2.0

    video = (video.clamp(0, 1) * 255).round().to(torch.uint8)  # [3,T,H,W]
    video = video.permute(1, 2, 3, 0)                          # [T,H,W,3]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    write_video(save_path, video, fps=int(fps))
    print(f"[save_video_tensor_as_mp4] saved to {save_path}")


# =========================
# 参数解析
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate video + flow from 32-ch VideoJAM WanX2.1-T2V-1.3B ckpt."
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model_videojam_32.safetensors",
        help="32 通道 VideoJAM ckpt 路径（safetensors）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./videojam_generate",
        help="输出视频保存目录",
    )

    # prompt 相关
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="指定一个 prompt；不指定则从 prompt_file 抽一行或用默认",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="/data/wubin/Self-Forcing-main/prompts/MovieGenVideoBench_extended.txt",
        help="从文件中随机抽一行当 prompt；如果文件不存在则用默认",
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="",
        help="negative prompt，可留空",
    )

    # 分辨率 / 帧数
    parser.add_argument(
        "--resolution",
        type=str,
        default="480p",
        help="分辨率标签，交给 get_image_size，例如 360p / 480p / 720p",
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default="16:9",
        help="宽高比，例如 16:9 / 9:16",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=80,
        help="生成的视频帧数 T（像 train_wanx2.1_t2v 里 preview_num_frames）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="输出视频帧率",
    )

    # scheduler / RF 超参
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="flow-matching 采样步数，对应 cfg.scheduler.sample_steps",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
        help="classifier-free guidance scale",
    )

    # 杂项
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="随机种子",
    )

    return parser.parse_args()


# =========================
# 主逻辑
# =========================

def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    assert torch.cuda.is_available(), "需要至少一块 GPU 做推理"
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---- minimal cfg，主要是给 scheduler.sample 用的 ----
    dtype = to_torch_dtype("bf16")
    max_seq_len = 75600  # 和训练里的 max_seq_len 对齐（你之前 config 写的）

    scheduler_cfg = dict(
        type="rflow-wanx",
        num_timesteps=1000,
        sample_steps=args.sample_steps,
        sample_shift=5.0,
        cfg_scale=args.cfg_scale,
        transform_scale=5.0,
        use_discrete_timesteps=False,
        sample_method="logit-normal",
        use_timestep_transform=True,
        use_fixed_timestep_transform=True,
    )

    cfg = Config(
        dict(
            dtype="bf16",
            max_seq_len=max_seq_len,
            scheduler=scheduler_cfg,
        )
    )

    # ==================================
    # 1. Text encoder (T5)
    # ==================================
    t5_cfg = dict(
        name="umt5_xxl",
        text_len=512,
        dtype="bf16",
        checkpoint_path="/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_path="/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl",
    )
    print("==> build T5 encoder ...")
    text_encoder = WanX21T5Encoder(
        name=t5_cfg["name"],
        text_len=t5_cfg["text_len"],
        dtype=t5_cfg["dtype"],
        device=device,
        checkpoint_path=t5_cfg["checkpoint_path"],
        tokenizer_path=t5_cfg["tokenizer_path"],
    )
    text_encoder.model = text_encoder.model.to(device=device).eval()

    # ==================================
    # 2. VAE
    # ==================================
    vae_cfg = dict(
        type="WanX21_VAE",
        vae_pth="/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE_for_wanx_code.pth",
    )
    print("==> build VAE ...")
    vae = build_module(vae_cfg, MODELS)
    vae.model = vae.model.to(device=device, dtype=dtype).eval()

    base_z_dim = vae.model.z_dim       # 一般是 16
    joint_z_dim = base_z_dim * 2       # 32 通道：video + flow
    print(f"[generate] base_z_dim={base_z_dim}, joint_z_dim={joint_z_dim}")

    # ==================================
    # 3. Scheduler (RF)
    # ==================================
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # ==================================
    # 4. 32-ch VideoJAM 模型
    # ==================================
    print("==> build 32-ch VideoJAM WanX model ...")
    model_cfg = dict(
        type="WanX21",
        patch_size=(1, 2, 2),
        text_len=t5_cfg["text_len"],
        in_dim=joint_z_dim,    # 32
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=4096,
        out_dim=joint_z_dim,   # 32
        num_heads=12,
        num_layers=30,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        use_fixed_seq_len=False,
        sp_degree=1,
    )
    model = build_module(model_cfg, MODELS)

    ckpt_path = args.ckpt
    assert os.path.exists(ckpt_path), f"ckpt not found: {ckpt_path}"
    print(f"==> load 32-ch VideoJAM ckpt from: {ckpt_path}")
    load_checkpoint(model, ckpt_path)
    print("    [done] VideoJAM model loaded.")

    model = model.to(device=device, dtype=dtype).eval()

    # ==================================
    # 5. 分辨率 / latent 尺寸
    # ==================================
    H, W = get_image_size(args.resolution, args.aspect_ratio)
    T = args.num_frames
    print(f"[generate] resolution={args.resolution}, aspect_ratio={args.aspect_ratio}, H={H}, W={W}, T={T}")

    latent_T = (T - 1) // vae.model.temporal_scale_factor + 1
    latent_H = H // vae.model.spatial_scale_factor
    latent_W = W // vae.model.spatial_scale_factor
    print(f"[generate] latent shape: [B=1, C={joint_z_dim}, T={latent_T}, H={latent_H}, W={latent_W}]")

    # ==================================
    # 6. 选 prompt
    # ==================================
    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_file is not None and os.path.exists(args.prompt_file):
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if lines:
            prompt = random.choice(lines)
        else:
            prompt = "A girl is running in a futuristic city at night."
    else:
        prompt = "一位少女在城市夜景中奔跑，镜头跟随，光影流动感强"

    print(f"==> prompt: {prompt}")
    neg_prompt = args.neg_prompt

    # ==================================
    # 7. RF 采样：直接在 32ch latent 上采样
    # ==================================
    with torch.no_grad():
        y = encode_prompt_inference(
            prompt=prompt,
            neg_prompt=neg_prompt,
            text_encoder=text_encoder,
            max_seq_len=max_seq_len,
        )

        model_args = {
            "height": torch.tensor([H], device=device, dtype=torch.float32),
            "width": torch.tensor([W], device=device, dtype=torch.float32),
            "num_frames": torch.tensor([T], device=device, dtype=torch.float32),
        }

        generator = torch.Generator(device=device).manual_seed(args.seed)

        z = torch.randn(
            1,
            joint_z_dim,
            latent_T,
            latent_H,
            latent_W,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        print("[generate] start RF sampling with 32-ch VideoJAM model ...")
        samples = scheduler.sample(
            model,
            y,
            z=z,
            prompts=[prompt],
            device=device,
            additional_args=model_args,
            progress=True,
            mask=None,
            generator=generator,
            cfg=cfg,
            mode="t2v",
        )

    # ==================================
    # 8. 拆成 video / flow latent，并分别 decode
    # ==================================
    joint_latent = samples  # [1, 32, T', H', W']
    video_latent = joint_latent[:, :base_z_dim]      # 前 16 通道：视频
    flow_latent  = joint_latent[:, base_z_dim:]      # 后 16 通道：光流

    print(f"[generate] video_latent.shape={video_latent.shape}, flow_latent.shape={flow_latent.shape}")

    with torch.no_grad():
        video_dec = vae.decode(video_latent)
        flow_dec  = vae.decode(flow_latent)

    if isinstance(video_dec, (list, tuple)):
        video_rec = video_dec[0]
    else:
        video_rec = video_dec

    if isinstance(flow_dec, (list, tuple)):
        flow_rec = flow_dec[0]
    else:
        flow_rec = flow_dec

    print(f"[generate] video_rec.shape={video_rec.shape}, flow_rec.shape={flow_rec.shape}")
    print(f"[generate] video_rec range=[{video_rec.min().item():.4f}, {video_rec.max().item():.4f}]")
    print(f"[generate] flow_rec  range=[{flow_rec.min().item():.4f}, {flow_rec.max().item():.4f}]")

    # ==================================
    # 9. 保存两个 mp4：视频 & “光流”
    # ==================================
    os.makedirs(args.output_dir, exist_ok=True)
    video_path = os.path.join(args.output_dir, "videojam_video.mp4")
    flow_path  = os.path.join(args.output_dir, "videojam_flow.mp4")

    save_video_tensor_as_mp4(video_rec, video_path, fps=args.fps)
    save_video_tensor_as_mp4(flow_rec,  flow_path,  fps=args.fps)

    print("[generate] done.")


if __name__ == "__main__":
    main()
