#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn.functional as F
from torchvision.io import read_video, write_video

from mmengine.config import Config

from vidgen.registry import MODELS, build_module
from vidgen.datasets.aspect import get_image_size
from vidgen.utils.misc import to_torch_dtype
from vidgen.utils.ckpt_utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="用原版 WanX 模型测试对光流视频的去噪能力（单输入单输出，不用文本）"
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="原版单分支 WanX 权重（diffusion_pytorch_model.safetensors 或 .pth）",
    )
    parser.add_argument(
        "--flow_path",
        type=str,
        required=True,
        help="光流视频路径（例如某个 flow.mp4）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./flow_denoise_test",
        help="输出视频保存目录",
    )

    # 噪声强度：用 t_index / num_timesteps 来控制
    parser.add_argument(
        "--t_index",
        type=int,
        default=500,
        help="噪声时间步 t（0~num_timesteps），噪声强度约为 t / num_timesteps，默认 500",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
        help="训练时使用的总时间步数（一般是 1000）",
    )

    # 空间/时间大小（会把 flow 视频 resize / 裁掉到这个配置）
    parser.add_argument(
        "--resolution",
        type=str,
        default="480p",
        help="分辨率标签（交给 get_image_size，例如 480p/720p）",
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default="9:16",
        help="宽高比，例如 16:9 / 9:16",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=80,
        help="使用的帧数（会从光流视频头部截断）",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="随机种子（用于噪声）",
    )

    return parser.parse_args()


def save_video_tensor_as_mp4(video: torch.Tensor, save_path: str, fps: int = 16):
    """
    video: [3, T, H, W] 或 [B,3,T,H,W]
    值范围 [-1,1] 或 [0,1]
    """
    if video.ndim == 5:
        video = video[0]
    assert video.ndim == 4, f"Expect [3,T,H,W], got {video.shape}"

    video = video.detach().cpu().float()

    # [-1,1] -> [0,1]
    if video.min() < 0.0:
        video = (video.clamp(-1, 1) + 1) / 2.0

    video = video.clamp(0, 1)
    video = (video * 255).round().to(torch.uint8)  # [3,T,H,W]
    video = video.permute(1, 2, 3, 0).contiguous()  # [T,H,W,3]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    write_video(save_path, video, fps=int(fps))
    print(f"[save_video_tensor_as_mp4] saved to {save_path}")


def load_and_preprocess_flow(path, device, dtype, target_H, target_W, num_frames):
    """
    读取光流 mp4，裁剪到指定帧数，并 resize 到 (target_H, target_W)，输出 [1,3,T,H,W]、[-1,1]
    """
    print(f"==> 读取光流视频: {path}")
    video, _, info = read_video(path, pts_unit="sec")  # [T, H, W, 3], uint8
    print(f"[raw flow] shape={video.shape}, fps={info.get('video_fps', 'N/A')}")

    # [T,H,W,3] -> [T,3,H,W]
    video = video.permute(0, 3, 1, 2).float() / 255.0  # [T,3,H,W], [0,1]

    # 截断帧数
    T = video.shape[0]
    if T > num_frames:
        video = video[:num_frames]
        T = num_frames
    print(f"[flow] after trunc to {T} frames, shape={video.shape}")

    # resize 到目标大小
    # 先 [T,3,H,W] -> [T,3,target_H,target_W]
    video = F.interpolate(
        video,
        size=(target_H, target_W),
        mode="bilinear",
        align_corners=False,
    )

    # [T,3,H,W] -> [1,3,T,H,W]
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,H,W]

    # [0,1] -> [-1,1]
    video = video * 2.0 - 1.0
    video = video.to(device=device, dtype=dtype)

    print(f"[flow] after resize and scale, shape={video.shape}, "
          f"range=({video.min().item():.3f},{video.max().item():.3f})")

    return video


def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    assert torch.cuda.is_available(), "需要 GPU 支持"
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    dtype = to_torch_dtype("bf16")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ===== 1. 解析空间尺寸 =====
    H, W = get_image_size(args.resolution, args.aspect_ratio)
    print(f"[配置] target resolution: H={H}, W={W}, num_frames={args.num_frames}")

    # ===== 2. 构建 VAE =====
    print("==> 初始化 WanX21 VAE ...")
    vae_cfg = dict(
        type="WanX21_VAE",
        vae_pth="/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE_for_wanx_code.pth",
    )
    vae = build_module(vae_cfg, MODELS)
    vae.model = vae.model.to(device=device, dtype=dtype).eval()
    base_z_dim = vae.model.z_dim  # 一般是 16
    print(f"[VAE] z_dim = {base_z_dim}")

    # ===== 3. 构建单分支 WanX 模型（in_dim=out_dim=z_dim） =====
    print("==> 构建单输入单输出 WanX 模型 ...")
    # 模型结构请根据你当前用的 13B NAS 配置来，下面是你之前用的那套
    model_cfg = dict(
        type="WanX21",
        patch_size=(1, 2, 2),
        text_len=512,      # 不用文本，但结构保持一致
        in_dim=base_z_dim,
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=4096,
        out_dim=base_z_dim,
        num_heads=12,
        num_layers=30,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        use_fixed_seq_len=False,
        sp_degree=1,
    )
    cfg = Config(dict(model=model_cfg))

    model = build_module(cfg.model, MODELS)
    model = model.to(device=device, dtype=dtype)

    print(f"==> 从权重加载: {args.ckpt_path}")
    assert os.path.exists(args.ckpt_path), f"权重文件不存在: {args.ckpt_path}"
    load_checkpoint(model, args.ckpt_path)
    model.eval()
    print("==> 模型权重加载完成")

    # ===== 4. 读取光流视频并编码到 latent =====
    flow_video = load_and_preprocess_flow(
        args.flow_path,
        device=device,
        dtype=dtype,
        target_H=H,
        target_W=W,
        num_frames=args.num_frames,
    )  # [1,3,T,H,W]

    with torch.no_grad():
        z_list = vae.encode(flow_video)   # 列表长度=B，每个 [z_dim,Tz,Hz,Wz]
        z_clean = torch.stack(z_list, dim=0)  # [B, z_dim, Tz, Hz, Wz]
    print(f"[latent] z_clean shape={z_clean.shape}, "
          f"range=({z_clean.min().item():.3f},{z_clean.max().item():.3f})")

    z_clean = z_clean.to(device=device, dtype=dtype)


    # ===== 5. 按 Rectified Flow 的公式加噪 =====
    num_timesteps = args.num_timesteps
    t_int = max(0, min(num_timesteps, args.t_index))
    sigma = float(t_int) / float(num_timesteps)
    print(f"[noise] 使用 t={t_int} / {num_timesteps} -> sigma={sigma:.3f}")

    # generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn_like(z_clean)

    # x_t = (1 - sigma) * x_0 + sigma * eps
    z_noisy = (1.0 - sigma) * z_clean + sigma * noise
    print(f"[latent] z_noisy shape={z_noisy.shape}, "
          f"range=({z_noisy.min().item():.3f},{z_noisy.max().item():.3f})")

    # ===== 6. 把 noisy latent 丢进模型，预测 v(x_t, t) ≈ eps - x_0 =====
    # 不用文本：构造一个全零的 dummy context，相当于“无条件”
    B = z_noisy.shape[0]
    text_dim = model_cfg["text_dim"]
    dummy_context = torch.zeros(
        B, 1, text_dim,
        device=device,
        dtype=dtype,
    )  # [B, 1, text_dim]
    seq_len = 1

    t_tensor = torch.full(
        (B,),
        t_int,
        device=device,
        dtype=torch.long,
    )  # [B], 和训练时一样使用 int64

    with torch.no_grad():
        # 模型前向：x_t, t, context, seq_len
        v_pred = model(
            z_noisy,
            t_tensor,
            context=dummy_context,
            seq_len=seq_len,
        )  # [B, z_dim, Tz, Hz, Wz]

    print(f"[latent] v_pred shape={v_pred.shape}, "
          f"range=({v_pred.min().item():.3f},{v_pred.max().item():.3f})")

    # 训练目标是 v ≈ eps - x_0，所以 x_0 ≈ eps - v
    z_denoised = noise.float() - v_pred.float()
    z_denoised = z_denoised.to(dtype)
    print(f"[latent] z_denoised shape={z_denoised.shape}, "
          f"range=({z_denoised.min().item():.3f},{z_denoised.max().item():.3f})")

    # ===== 7. VAE 解码，分别得到：干净 flow / 加噪 flow / 去噪 flow =====
    with torch.no_grad():
        flow_clean_dec = vae.decode(z_clean.to(dtype))
        flow_noisy_dec = vae.decode(z_noisy.to(dtype))
        flow_denoised_dec = vae.decode(z_denoised.to(dtype))

    def _first(x):
        return x[0] if isinstance(x, (list, tuple)) else x

    flow_clean = _first(flow_clean_dec)     # [B,3,T,H,W] 或 [3,T,H,W]
    flow_noisy = _first(flow_noisy_dec)
    flow_deno = _first(flow_denoised_dec)

    # 保证形状 [3,T,H,W]
    if flow_clean.ndim == 5:
        flow_clean = flow_clean[0]
    if flow_noisy.ndim == 5:
        flow_noisy = flow_noisy[0]
    if flow_deno.ndim == 5:
        flow_deno = flow_deno[0]

    print(f"[decode] clean range=({flow_clean.min().item():.3f},{flow_clean.max().item():.3f})")
    print(f"[decode] noisy range=({flow_noisy.min().item():.3f},{flow_noisy.max().item():.3f})")
    print(f"[decode] deno range=({flow_deno.min().item():.3f},{flow_deno.max().item():.3f})")

    # ===== 8. 保存三个视频 =====
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.flow_path))[0]
    clean_path = os.path.join(args.output_dir, f"{base_name}_clean.mp4")
    noisy_path = os.path.join(args.output_dir, f"{base_name}_noisy_t{t_int}.mp4")
    deno_path  = os.path.join(args.output_dir, f"{base_name}_denoised_t{t_int}.mp4")

    fps = 16  # 你可以根据实际 flow fps 调整
    save_video_tensor_as_mp4(flow_clean, clean_path, fps=fps)
    save_video_tensor_as_mp4(flow_noisy, noisy_path, fps=fps)
    save_video_tensor_as_mp4(flow_deno, deno_path, fps=fps)

    print("==> 去噪实验完成！")
    print(f"    原始 flow:  {clean_path}")
    print(f"    加噪 flow:  {noisy_path}")
    print(f"    去噪结果:   {deno_path}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
