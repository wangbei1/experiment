#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import random
import json
import torch
from torchvision.io import write_video
from mmengine.config import Config

from vidgen.registry import MODELS, SCHEDULERS, build_module
from vidgen.models.text_encoder import WanX21T5Encoder
from vidgen.datasets.aspect import get_image_size
from vidgen.utils.misc import to_torch_dtype
from vidgen.utils.ckpt_utils import load_checkpoint


# =========================
# 从训练 distcp 检查点加载模型
# =========================

def load_training_checkpoint(checkpoint_dir, device, model, ema_model=None,
                             optimizer=None, lr_scheduler=None, sampler=None, mode='all'):
    """
    从分布式保存的 checkpoint 目录加载完整状态（单机单卡推理用）
    checkpoint_dir: 形如  outputs_xxx/000-xxx/  的根目录
      其下包含  epochX-global_stepY 子目录，每个目录下面有 all/__0_0.distcp + .metadata + running_states.json
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"检查点目录不存在: {checkpoint_dir}")

    # 列出 epoch 子目录
    checkpoint_dirs = [
        d for d in os.listdir(checkpoint_dir)
        if d.startswith('epoch') and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not checkpoint_dirs:
        raise ValueError(f"在 {checkpoint_dir} 中未找到有效的检查点")

    # 解析 epoch、global_step 并按最新排序
    def _parse_epoch_gs(name):
        # 形如 epoch0-global_step400
        parts = name.split('-')
        ep = int(parts[0].replace('epoch', ''))
        gs = int(parts[1].replace('global_step', ''))
        return ep, gs

    checkpoint_dirs.sort(key=lambda x: _parse_epoch_gs(x), reverse=True)
    latest_dir_name = checkpoint_dirs[0]
    latest_checkpoint = os.path.join(checkpoint_dir, latest_dir_name)
    print(f"==> 加载最新检查点: {latest_checkpoint}")

    # 运行状态
    running_states_path = os.path.join(latest_checkpoint, "running_states.json")
    if os.path.exists(running_states_path):
        with open(running_states_path, 'r') as f:
            running_states = json.load(f)
        epoch = running_states.get("epoch", 0)
        global_step = running_states.get("global_step", 0)
        print(f"==> 检查点状态: epoch={epoch}, global_step={global_step}")
    else:
        epoch = 0
        global_step = 0
        print("==> 未找到 running_states.json，使用默认状态")

    # distcp 存放位置，比如  .../epoch0-global_step400/all
    checkpoint_path = os.path.join(latest_checkpoint, mode)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点路径不存在: {checkpoint_path}")

    # 使用 torch.distributed.checkpoint.load 合并分片
    from torch.distributed.checkpoint import load

    full_state_dict = {}
    load(full_state_dict, checkpoint_id=checkpoint_path)

    # 提取并加载
    if 'model' in full_state_dict:
        model.load_state_dict(full_state_dict['model'])
        print("==> 模型权重加载完成")

    if ema_model is not None and 'ema' in full_state_dict:
        ema_model.load_state_dict(full_state_dict['ema'])
        print("==> EMA 模型权重加载完成")

    if optimizer is not None and 'optimizer' in full_state_dict:
        optimizer.load_state_dict(full_state_dict['optimizer'])
        print("==> 优化器状态加载完成")

    return epoch, global_step, latest_dir_name


# =========================
# 工具函数
# =========================

def encode_prompt_inference(prompt, neg_prompt, text_encoder, max_seq_len):
    """编码提示词（条件 + 无条件），供 RF / CFG 使用。"""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    neg_prompt = [neg_prompt] if isinstance(neg_prompt, str) else neg_prompt

    context = text_encoder(prompt)          # 条件
    context_null = text_encoder(neg_prompt) # 无条件

    return dict(context=context, context_null=context_null, max_seq_len=max_seq_len)


def save_video_tensor_as_mp4(video: torch.Tensor, save_path: str, fps: int = 24):
    """保存视频张量为 MP4 文件"""
    if video.ndim == 5:  # [B, 3, T, H, W]
        video = video[0]
    assert video.ndim == 4, f"Expect [3,T,H,W], got {video.shape}"

    video = video.detach().cpu().float()

    # [-1,1] -> [0,1]
    if video.min() < 0.0:
        video = (video.clamp(-1, 1) + 1) / 2.0

    video = video.clamp(0, 1)
    video = (video * 255).round().to(torch.uint8)  # [3,T,H,W]
    video = video.permute(1, 2, 3, 0).contiguous() # [T,H,W,3]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    write_video(save_path, video, fps=int(fps))
    print(f"[save_video_tensor_as_mp4] saved to {save_path}")


# =========================
# 参数解析
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="VideoJAM（video + flow，token concat）推理：从训练检查点或预训练权重加载并生成视频+流"
    )

    # 只能二选一：预训练权重 or 训练检查点目录
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument(
        "--ckpt_path",
        type=str,
        help="预训练权重路径（.safetensors 或 .pth，单文件，对应 WanX21VideoJAM）"
    )
    ckpt_group.add_argument(
        "--checkpoint_dir",
        type=str,
        help="训练检查点目录（包含 epochX-global_stepY 子目录，distcp 格式）"
    )

    # 检查点模式（只对 checkpoint_dir 有效）
    parser.add_argument(
        "--checkpoint_mode",
        type=str,
        choices=['all'],
        default='all',
        help="distcp 模式，一般保持 all 即可"
    )

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="如果从训练检查点加载，并且其中包含 ema，则使用 ema 模型推理"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./videojam_generate",
        help="输出视频保存目录"
    )

    # prompt 相关
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="指定一个 prompt；不指定则从 prompt_file 随机抽一行"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="/data/wubin/Self-Forcing-main/prompts/MovieGenVideoBench_extended.txt",
        help="从文件中随机抽一行当 prompt"
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="",
        help="negative prompt"
    )

    # 分辨率 / 帧数
    parser.add_argument(
        "--resolution",
        type=str,
        default="480p",
        help="分辨率标签（交给 get_image_size，例如 480p/720p）"
    )
    parser.add_argument(
        "--aspect_ratio",
        type=str,
        default="9:16",
        help="宽高比，例如 16:9 / 9:16"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=80,
        help="生成的视频帧数"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="输出视频帧率"
    )

    # scheduler 超参
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="flow-matching/RF 采样步数"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
        help="classifier-free guidance scale"
    )

    # 杂项
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="随机种子"
    )

    return parser.parse_args()


# =========================
# 主逻辑（VideoJAM 双模态）
# =========================

def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    assert torch.cuda.is_available(), "需要 GPU 支持"
    device = torch.device("cuda")
    torch.cuda.set_device(0)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---- 基础配置 ----
    dtype = to_torch_dtype("bf16")
    max_seq_len = 75600

    # 这里只给 scheduler 必要字段，其他从 args 里取
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
    cfg = Config(dict(
        dtype="bf16",
        max_seq_len=max_seq_len,
        scheduler=scheduler_cfg,
    ))

    # ==================================
    # 1. 文本编码器（T5）
    # ==================================
    print("==> 初始化 T5 文本编码器 ...")
    t5_cfg = dict(
        name="umt5_xxl",
        text_len=512,
        dtype="bf16",
        checkpoint_path="/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_path="/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl",
    )

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
    print("==> 初始化 VAE ...")
    vae_cfg = dict(
        type="WanX21_VAE",
        vae_pth="/data/wubin/wanx-code/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE_for_wanx_code.pth",
    )
    vae = build_module(vae_cfg, MODELS)
    vae.model = vae.model.to(device=device, dtype=dtype).eval()

    base_z_dim = vae.model.z_dim          # 单分支 latent 通道数（例如 16）
    print(f"[模型配置] base_z_dim={base_z_dim} (单分支 latent，每个模态一份)")

    # ==================================
    # 3. RF 调度器（外层封装）
    # ==================================
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    flow_solver = scheduler.scheduler  # FlowDPMSolverMultistepScheduler 实例

    # ==================================
    # 4. 构建 WanX21VideoJAM 模型（token 拼接双模态）
    # ==================================
    print("==> 构建 VideoJAM 模型 (in_dim = out_dim = base_z_dim，num_modalities=2) ...")
    model_cfg = dict(
        type="WanX21",    # ⭐ 使用你新注册的 VideoJAM 版本
        patch_size=(1, 2, 2),
        text_len=t5_cfg["text_len"],
        in_dim=base_z_dim,        # 每个模态 16 通道
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=4096,
        out_dim=base_z_dim,       # 每个模态输出 16 通道
        num_heads=12,
        num_layers=30,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        use_fixed_seq_len=False,
        sp_degree=1,              # 推理阶段一般不开 SP，保持 1 即可
        num_modalities=2,         # ⭐ 关键：双模态
    )
    model = build_module(model_cfg, MODELS)
    model = model.to(device=device, dtype=dtype)

    # 如果需要从 checkpoint_dir 使用 EMA，就再构建一个同结构模型
    ema_model = None
    if args.use_ema and args.checkpoint_dir is not None:
        ema_model = build_module(model_cfg, MODELS).to(device=device, dtype=dtype)

    # ==================================
    # 5. 加载权重
    # ==================================
    latest_subdir_name = "pretrained"
    if args.checkpoint_dir:
        print("==> 从训练检查点加载模型 ...")
        epoch, global_step, latest_subdir_name = load_training_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            device=device,
            model=model,
            ema_model=ema_model if args.use_ema else None,
            mode=args.checkpoint_mode,
        )
        print(f"==> 成功加载检查点: epoch={epoch}, global_step={global_step}")
    elif args.ckpt_path:
        print(f"==> 从预训练权重加载: {args.ckpt_path}")
        assert os.path.exists(args.ckpt_path), f"权重文件不存在: {args.ckpt_path}"
        load_checkpoint(model, args.ckpt_path)
        print("==> 预训练权重加载完成")
        if args.use_ema:
            print("⚠️ 提示: 当前 ckpt_path 不包含单独 ema 权重，--use_ema 标志将被忽略。")

    # 选择推理模型
    if args.use_ema and ema_model is not None and args.checkpoint_dir:
        print("==> 使用 EMA 模型进行推理")
        inference_model = ema_model
    else:
        print("==> 使用普通模型进行推理")
        inference_model = model

    inference_model.eval()

    # ==================================
    # 6. 分辨率配置
    # ==================================
    H, W = get_image_size(args.resolution, args.aspect_ratio)
    T = args.num_frames
    print(f"[推理配置] 分辨率: {args.resolution}, 宽高比: {args.aspect_ratio}")
    print(f"[推理配置] 视频尺寸: H={H}, W={W}, T={T}")

    latent_T = (T - 1) // vae.model.temporal_scale_factor + 1
    latent_H = H // vae.model.spatial_scale_factor
    latent_W = W // vae.model.spatial_scale_factor
    print(f"[推理配置] 潜在空间尺寸: T={latent_T}, H={latent_H}, W={latent_W}")

    # ==================================
    # 7. 处理 prompt
    # ==================================
    if args.prompt is not None:
        prompt = args.prompt
    elif args.prompt_file is not None and os.path.exists(args.prompt_file):
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        prompt = random.choice(lines) if lines else "A girl is running in a futuristic city at night."
    else:
        prompt = "一位少女在城市夜景中奔跑，镜头跟随，光影流动感强"

    print(f"==> 提示词: {prompt}")
    neg_prompt = args.neg_prompt

    # ==================================
    # 8. RF 采样（VideoJAM：video + flow，token concat）
    # ==================================
    print("==> 开始 VideoJAM 双模态视频生成推理 ...")

    with torch.no_grad():
        # 编码文本（拿到 context / context_null / max_seq_len）
        y = encode_prompt_inference(
            prompt=prompt,
            neg_prompt=neg_prompt,
            text_encoder=text_encoder,
            max_seq_len=max_seq_len,
        )

        # 多模态 sample_videojam 只用到 context/context_null/max_seq_len
        model_args = {
            "context": y["context"],
            "context_null": y["context_null"],
            "max_seq_len": y["max_seq_len"],
        }

        # 初始噪声 latent：
        #   video: [1, base_z_dim, T', H', W']
        #   flow:  [1, base_z_dim, T', H', W']
        #   concat -> z: [1, 2*base_z_dim, T', H', W']
        generator = torch.Generator(device=device).manual_seed(args.seed)
        z_video = torch.randn(
            1,
            base_z_dim,
            latent_T,
            latent_H,
            latent_W,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        z_flow = torch.randn_like(z_video)
        z = torch.cat([z_video, z_flow], dim=1)  # [1, 2C, T', H', W']

        print("==> 进行 Rectified Flow 采样 (VideoJAM: video + flow，token concat) ...")

        # 调用你在 FlowDPMSolverMultistepScheduler 里实现的 sample_videojam
        samples = flow_solver.sample_videojam(
            model=inference_model,
            z=z,
            model_kwargs=model_args,
            device=device,
            sample_steps=args.sample_steps,
            sample_shift=scheduler_cfg["sample_shift"],
            guidance_scale=args.cfg_scale,
            generator=generator,
            mode="t2v",
            progress=True,
        )

    # ==================================
    # 9. 解码 & 保存（video + flow 两个结果）
    # ==================================
    print("==> 解码生成结果 ...")
    latent = samples  # [1, 2*base_z_dim, T', H', W']

    video_latent = latent[:, :base_z_dim]
    flow_latent  = latent[:, base_z_dim:]

    with torch.no_grad():
        video_dec = vae.decode(video_latent)
        flow_dec  = vae.decode(flow_latent)

    video_rec = video_dec[0] if isinstance(video_dec, (list, tuple)) else video_dec
    flow_rec  = flow_dec[0]  if isinstance(flow_dec,  (list, tuple)) else flow_dec

    print(f"[生成结果] video 范围: [{video_rec.min().item():.3f}, {video_rec.max().item():.3f}]")
    print(f"[生成结果] flow  范围: [{flow_rec.min().item():.3f}, {flow_rec.max().item():.3f}]")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.checkpoint_dir:
        tag = latest_subdir_name  # epochX-global_stepY
    else:
        tag = "pretrained"

    video_path = os.path.join(args.output_dir, f"videojam_{tag}_video.mp4")
    flow_path  = os.path.join(args.output_dir, f"videojam_{tag}_flow.mp4")

    save_video_tensor_as_mp4(video_rec, video_path, fps=args.fps)
    save_video_tensor_as_mp4(flow_rec,  flow_path,  fps=args.fps)

    print("==> 推理完成！")
    print(f"==> 视频保存至: {video_path}")
    print(f"==> Flow 保存至:  {flow_path}")


if __name__ == "__main__":
    main()
