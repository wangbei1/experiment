#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random
import copy

import torch
from mmengine.config import Config

from vidgen.registry import MODELS, SCHEDULERS, build_module
from vidgen.models.text_encoder import WanX21T5Encoder
from vidgen.utils.ckpt_utils import load_checkpoint
from vidgen.utils.misc import to_torch_dtype


# ---------------------------
# 文本编码工具
# ---------------------------

def encode_prompt_inference(
    prompt,
    neg_prompt,
    text_encoder,
    max_seq_len,
):
    """
    和 train_wanx2.1_t2v.py 里的一样：返回 context / context_null / max_seq_len
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt
    neg_prompt = [neg_prompt] if isinstance(neg_prompt, str) else neg_prompt

    context = text_encoder(prompt)          # 条件分支
    context_null = text_encoder(neg_prompt) # 无条件分支

    return dict(context=context, context_null=context_null, max_seq_len=max_seq_len)


# ---------------------------
# 参数解析
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug forward consistency between base (16ch) WanX and expanded (32ch) VideoJAM model."
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
        help="从文件中随机抽一行当 prompt；如果文件不存在则用默认 prompt",
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="",
        help="negative prompt，可留空",
    )

    # debug 步数：在多少个不同的 t 上做 forward 对比
    parser.add_argument(
        "--num_debug_steps",
        type=int,
        default=50,
        help="在多少个 time steps 上比较 base vs videojam 的 forward 输出",
    )

    # 可选：你导出的 32 通道 videojam ckpt
    parser.add_argument(
        "--videojam_ckpt",
        type=str,
        help="(可选) 你导出的 32ch VideoJAM ckpt；为空则只用 expand_to_videojam 后的权重",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="随机种子（控制噪声和 prompt 采样）",
    )

    return parser.parse_args()


# ---------------------------
# 主逻辑：forward 对比调试
# ---------------------------

def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    assert torch.cuda.is_available(), "需要至少一块 GPU 做调试"
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    # ====== 基础 cfg（只写 forward 需要的） ======
    cfg = Config(
        dict(
            dtype="bf16",
            max_seq_len=75600,  # 和你训练里一致
            t5=dict(
                name="umt5_xxl",
                text_len=512,
                dtype="bf16",
                checkpoint_path="/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                tokenizer_path="/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl",
            ),
        )
    )
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ====== 1. 文本编码器 ======
    print("==> build T5 encoder ...")
    t5_cfg = cfg.t5
    text_encoder = WanX21T5Encoder(
        name=t5_cfg.name,
        text_len=t5_cfg.text_len,
        dtype=t5_cfg.dtype,
        device=device,
        checkpoint_path=t5_cfg.checkpoint_path,
        tokenizer_path=t5_cfg.tokenizer_path,
    )
    text_encoder.model = text_encoder.model.to(device=device).eval()

    # ====== 2. 选一个 prompt 并 encode，拿到 context / max_seq_len ======
    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_file is not None and os.path.exists(args.prompt_file):
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        prompt = random.choice(lines) if lines else "A girl is running in a futuristic city at night."
    else:
        prompt = "一位少女在城市夜景中奔跑，镜头跟随，光影流动感强"

    print(f"==> prompt: {prompt}")
    neg_prompt = args.neg_prompt

    with torch.no_grad():
        y = encode_prompt_inference(
            prompt=prompt,
            neg_prompt=neg_prompt,
            text_encoder=text_encoder,
            max_seq_len=cfg.max_seq_len,
        )
    context = y["context"]      # [B, L, D]
    max_seq_len = y["max_seq_len"]

    # ====== 3. 构建 base 16ch WanX 模型 ======
    base_z_dim = 16
    joint_z_dim = base_z_dim * 2

    print("==> build base 16ch WanX model")
    model_cfg_base = dict(
        type="WanX21",
        patch_size=(1, 2, 2),
        text_len=t5_cfg.text_len,
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
    base_model = build_module(model_cfg_base, MODELS).to(device=device, dtype=dtype)

    diffusion_ckpt = "/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
    print("init wanx model by random")
    print(f"==> load base ckpt from: {diffusion_ckpt}")
    load_checkpoint(base_model, diffusion_ckpt)
    base_model.eval()
    print("    [done] base_model loaded.")

    # ====== 4. deepcopy + expand_to_videojam ======
    print("==> deepcopy base_model -> videojam_model & expand_to_videojam(base_channels=16)")
    vj_model = copy.deepcopy(base_model)
    vj_model.expand_to_videojam(base_channels=base_z_dim)
    print(
        f"[VideoJAM] Expanded Transformer: in_dim {base_model.in_dim} -> {vj_model.in_dim}, "
        f"out_dim {base_model.out_dim} -> {vj_model.out_dim}"
    )

    # 可选：加载你导出的 32ch VideoJAM ckpt（如果是从这个 vj_model.state_dict() 保存出来的）
    if args.videojam_ckpt and os.path.exists(args.videojam_ckpt):
        print(f"==> load 32ch videojam ckpt from: {args.videojam_ckpt}")
        load_checkpoint(vj_model, args.videojam_ckpt)
        print("    [videojam_ckpt] loaded.")
    else:
        if args.videojam_ckpt:
            print(f"[WARN] videojam_ckpt path not found: {args.videojam_ckpt}, 仅使用 expand_to_videojam 后的权重。")

    vj_model = vj_model.to(device=device, dtype=dtype).eval()

    # ====== 5. forward 一致性调试 ======
    num_debug_steps = max(1, args.num_debug_steps)
    print(f"==> use fake latent shape for debug ...")
    # 选一个合法的 latent 尺寸（与 patch_size=(1,2,2) 对齐即可）
    Tz, Hz, Wz = 16, 60, 104
    print(f"    fake latent shape: [B=1, C=16, T={Tz}, H={Hz}, W={Wz}]")

    g = torch.Generator(device=device).manual_seed(args.seed)

    # 基础 16 通道噪声（video latent）
    z16 = torch.randn(
        1, base_z_dim, Tz, Hz, Wz,
        device=device, dtype=dtype, generator=g
    )
    # 扩展到 32 通道：后 16 通道全 0（flow 分支）
    z32 = torch.cat([z16, torch.randn_like(z16)], dim=1)

    # 5.1 先检查 patch_embedding 一致性
    with torch.no_grad():
        y_base = base_model.patch_embedding(z16)
        y_vj   = vj_model.patch_embedding(z32)

    print("---- patch_embedding outputs ----")
    print(
        f"y_base: shape={tuple(y_base.shape)}, "
        f"min={y_base.min().item():.4f}, max={y_base.max().item():.4f}, std={y_base.std().item():.4f}"
    )
    print(
        f"y_vj  : shape={tuple(y_vj.shape)},   "
        f"min={y_vj.min().item():.4f}, max={y_vj.max().item():.4f}, std={y_vj.std().item():.4f}"
    )
    diff_patch = (y_base - y_vj).abs()
    print(
        f"[patch diff] max={diff_patch.max().item():.6e}, "
        f"mean={diff_patch.mean().item():.6e}"
    )

    # 5.2 多个时间步上的 forward 对比
    num_train_timesteps = 1000  # rectified_flow 默认 1000
    t_list = torch.linspace(
        0, num_train_timesteps - 1,
        steps=num_debug_steps,
        device=device,
        dtype=torch.long,
    )

    print("==> start debugging forward consistency for multiple t steps ...")

    for i, t_scalar in enumerate(t_list):
        t_batch = t_scalar.view(1)  # [B], int64

        with torch.no_grad():
            # **关键：严格对齐 rectified_flow.training_losses 的调用方式**
            # arg_c = {'context': context, 'seq_len': max_seq_len}
            # model(x_t, t, **arg_c)
            out_base = base_model(z16, t_batch, context=context, seq_len=max_seq_len)  # [1, 16, ...]
            out_vj   = vj_model(z32, t_batch, context=context, seq_len=max_seq_len)    # [1, 32, ...]

        # 拆出 video / flow 两个分支
        out_vj_video = out_vj[:, :base_z_dim]
        out_vj_flow  = out_vj[:, base_z_dim:]

        diff_video = (out_vj_video - out_base).abs()
        video_max  = diff_video.max().item()
        video_mean = diff_video.mean().item()
        flow_std   = out_vj_flow.std().item()
        flow_max   = out_vj_flow.abs().max().item()

        print(
            f"[step {i+1:02d}/{num_debug_steps:02d}] "
            f"t={int(t_scalar.item()):4d} | "
            f"video_diff_max={video_max:.6e}, video_diff_mean={video_mean:.6e} | "
            f"flow_std={flow_std:.6e}, flow_max_abs={flow_max:.6e}"
        )

    print("==> debug forward done. (不做采样，只做 forward 一致性检查)")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
