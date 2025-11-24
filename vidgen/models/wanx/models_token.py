import os
import torch
import torch.nn as nn
import torch.amp as amp
import math
import torch.nn.functional as F
from .attention import flash_attention
import torch.distributed as dist
from vidgen.registry import MODELS
from vidgen.utils.ckpt_utils import load_checkpoint
from vidgen.acceleration.checkpoint import auto_grad_checkpoint

from training_acc.dist.parallel_state import is_enable_sequence_parallel
from training_acc.patches.wanx2_1_t2v import split, gather, collect_tokens, collect_heads

from .models import Transformer

__all__ = ['Transformer']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs, num_modalities: int = 1):
    """
    x:              [B, L, n_heads, head_dim]
    grid_sizes:     [B, 3] 每个样本的 (T', H', W')
    freqs:          [max_seq, head_dim/2] 预先生成的频率
    num_modalities: 1 = 单模态 (原逻辑)
                    2 = 双模态，假设序列是 [video_tokens, flow_tokens]，
                        两段长度相同，且共享同一套 (t,h,w) RoPE，相位一致
    """
    n, c = x.size(2), x.size(3) // 2  # c = head_dim / 2 (复数维)

    # split freqs 为 T / H / W 三段
    freqs_t, freqs_h, freqs_w = freqs.split(
        [c - 2 * (c // 3), c // 3, c // 3], dim=1
    )

    output = []
    B, L = x.shape[0], x.shape[1]

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len_per = f * h * w                 # 每个模态的 token 数
        total_len = seq_len_per * num_modalities  # video+flow 的总 token 数
        total_len = min(total_len, L)           # 防御一下越界

        # 只对前 total_len 个 token 做 RoPE，后面的（padding）不变
        x_main = x[i, :total_len].to(torch.float64)  # [total_len, n, 2c]
        x_complex = torch.view_as_complex(
            x_main.reshape(total_len, n, -1, 2)
        )  # [total_len, n, c]

        # 构造一套基础 (t,h,w) 的频率表 [seq_len_per, 1, c]
        base_freqs = torch.cat([
            freqs_t[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len_per, 1, -1)  # [seq_len_per, 1, c]

        if num_modalities == 1:
            freqs_i = base_freqs  # [seq_len_per, 1, c]
        else:
            # 把同一套 (t,h,w) 相位复制给每个模态：
            # [seq_len_per * num_modalities, 1, c]
            freqs_i = base_freqs.repeat(num_modalities, 1, 1)

        freqs_i = freqs_i[:total_len]  # 防御一下

        # 应用 RoPE：复数乘法 = 二维旋转
        x_rot = torch.view_as_real(x_complex * freqs_i).flatten(2)  # [total_len, n, 2c]

        # 把没处理的 padding token 拼回去
        if total_len < L:
            x_rest = x[i, total_len:]
            x_out = torch.cat([x_rot, x_rest], dim=0)
        else:
            x_out = x_rot

        output.append(x_out)

    return torch.stack(output).to(x.dtype)


class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class LayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)
    
    def forward(self, x):
        return super().forward(x)


class SelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(self, x, seq_lens, grid_sizes, freqs, num_modalities: int = 1):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        
        q, k, v = qkv_fn(x)

        if is_enable_sequence_parallel():
            q = collect_tokens(q)
            k = collect_tokens(k)
            v = collect_tokens(v)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs, num_modalities=num_modalities),
            k=rope_apply(k, grid_sizes, freqs, num_modalities=num_modalities),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size
        )

        if is_enable_sequence_parallel():
            x = collect_heads(x)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CrossAttention(SelfAttention):

    def forward(self, x, context, context_lens):
        """
        x:              [B, L1, C].
        context:        [B, L2, C].
        context_lens:   [B].
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        if is_enable_sequence_parallel():
            q = collect_tokens(q)
            k = split(k, 2)
            v = split(v, 2)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        if is_enable_sequence_parallel():
            x = collect_heads(x)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class AttentionBlock(nn.Module):
    
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = LayerNorm(dim, eps)
        self.self_attn = SelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = LayerNorm(
            dim, eps, elementwise_affine=True
        ) if cross_attn_norm else nn.Identity()
        self.cross_attn = CrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = LayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
    
    # 没有设置环境变量或者enable是false的时候，disable是true，即不启用
    @torch.compile(disable=os.getenv('ENABLE_COMPILE', 'FALSE').lower() == 'false')
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        num_modalities: int = 1,
    ):
        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0],
            seq_lens,
            grid_sizes,
            freqs,
            num_modalities=num_modalities
        )
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim_proj = math.prod(patch_size) * out_dim
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim_proj)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)
    
    def forward(self, x, e):
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class Transformer(nn.Module):

    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        use_fixed_seq_len=True,
        sp_degree=1,
        num_modalities: int = 1,      # 1: 原始单模态；2: video+flow 双模态
    ):
        super().__init__()
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_fixed_seq_len = use_fixed_seq_len
        self.sp_degree = sp_degree
        self.num_modalities = num_modalities

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )

        # modality embeddings（用于区分 video / flow）
        if self.num_modalities > 1:
            self.mod_embeddings = nn.Parameter(
                torch.randn(self.num_modalities, dim) / dim ** 0.5
            )
        else:
            self.mod_embeddings = None

        # blocks
        self.blocks = nn.ModuleList([AttentionBlock(
            dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps
        ) for _ in range(num_layers)])

        # head（每个模态输出 out_dim 通道）
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self._freqs_initialized = False

    # 原来的 expand_to_videojam 可以保留（不再强依赖），这里就不动了
    def expand_to_videojam(self, base_channels: int = 16):
        ...

    @property
    def freqs(self):
        if not self._freqs_initialized or not hasattr(self, '_freqs'):
            # 动态初始化保障逻辑
            d = self.dim // self.num_heads
            device = self.patch_embedding.weight.device
            
            # 确保在CUDA设备生成张量
            with torch.cuda.device(device):
                self._freqs = torch.cat([
                    rope_params(1024, d-4*(d//6)),
                    rope_params(1024, 2*(d//6)),
                    rope_params(1024, 2*(d//6))
                ], dim=1).to(device)
                
            self._freqs_initialized = True
        return self._freqs
    
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        flow=None,          # 如果提供 flow，则走双模态分支
    ):
        """
        单模态:
            x:        [B, C, T, H, W]
        双模态:
            x:        [B, C, T, H, W]  (video)
            flow:     [B, C, T, H, W]  (flow)
        t:            [B]
        context:      一个 list / tensor，每个元素 [L, C_text]
        seq_len:      最大 token 长度（通常 = T'*H'*W' 或 2*T'*H'*W'）
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self._freqs = self.freqs.to(device)

        # -------- 统一 padding（对 video / flow 一起） --------
        T, ori_height, ori_width = x.shape[-3:]
        pad_w = (self.patch_size[2] - ori_width % self.patch_size[2]) % self.patch_size[2]
        pad_h = (self.patch_size[1] - ori_height % self.patch_size[1]) % self.patch_size[1]

        if pad_w != 0 or pad_h != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            if flow is not None:
                flow = F.pad(flow, (0, pad_w, 0, pad_h))

        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # -------- time embeddings --------
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(x.dtype)
        )  # [B, dim]
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # [B, 6, dim]

        # -------- context --------
        context_lens = None
        context = self.text_embedding(torch.stack([torch.cat([
            u, u.new_zeros(self.text_len - u.size(0), u.size(1))
        ]) for u in context]))

        # ========== 单模态路径：保持原有行为 ==========
        if self.num_modalities == 1 or flow is None:
            # embeddings
            x = self.patch_embedding(x)  # [B, D, T', H', W']
            grid_sizes = torch.stack([
                torch.tensor(u.shape[1:], dtype=torch.long) for u in x
            ])
            x = x.flatten(2).transpose(1, 2)  # [B, S, D]
            seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)

            tokens_num = x.shape[1]
            remainder = 0
            if self.use_fixed_seq_len:
                # use fixed seq length
                assert tokens_num <= seq_len, f"{seq_len=}, {x[0].shape=}"
                padding_num = seq_len - tokens_num
                if padding_num > 0:
                    x = torch.cat(
                        [x, x.new_zeros(x.shape[0], padding_num, x.shape[2])],
                        dim=1
                    )
            else:
                remainder = tokens_num % int(self.sp_degree)  # 兼容sp
                if remainder != 0:
                    padding_num = self.sp_degree - remainder
                    x = torch.cat(
                        [x, x.new_zeros(x.shape[0], padding_num, x.shape[2])],
                        dim=1
                    )

            if is_enable_sequence_parallel():
                x = split(x, 1)

            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,
                num_modalities=1,
            )

            for block in self.blocks:
                x = auto_grad_checkpoint(block, x, **kwargs)
            
            # head
            x = self.head(x, e)

            if is_enable_sequence_parallel():
                x = gather(x, 1)

            if remainder != 0:
                x = x[:, :-padding_num]
                
            # unpatchify
            x = self.unpatchify(x, tt, th, tw)
            x = x[:, :, :, :ori_height, :ori_width]
            return x

        # ========== 双模态路径：video + flow ==========
        # 1) patch embedding 分别处理 video / flow
        x_v = self.patch_embedding(x)         # [B, D, T', H', W']
        x_f = self.patch_embedding(flow)      # [B, D, T', H', W']

        grid_sizes = torch.stack([
            torch.tensor(u.shape[1:], dtype=torch.long) for u in x_v
        ])  # 每个样本一行 [T', H', W']

        x_v = x_v.flatten(2).transpose(1, 2)  # [B, S, D]
        x_f = x_f.flatten(2).transpose(1, 2)  # [B, S, D]

        B, S, D = x_v.shape
        tokens_per_modality = S

        # 2) 加模态 embedding（区分 video / flow）
        if self.mod_embeddings is not None:
            x_v = x_v + self.mod_embeddings[0].view(1, 1, -1)
            x_f = x_f + self.mod_embeddings[1].view(1, 1, -1)

        # 3) concat 成一个长序列：[video_tokens, flow_tokens]
        x = torch.cat([x_v, x_f], dim=1)      # [B, 2S, D]
        tokens_num_total = x.shape[1]

        seq_lens = torch.full(
            (B,),
            tokens_num_total,
            dtype=torch.long
        )

        remainder = 0
        if self.use_fixed_seq_len:
            assert tokens_num_total <= seq_len, f"{seq_len=}, {tokens_num_total=}"
            padding_num = seq_len - tokens_num_total
            if padding_num > 0:
                x = torch.cat(
                    [x, x.new_zeros(B, padding_num, D)],
                    dim=1
                )
        else:
            remainder = tokens_num_total % int(self.sp_degree)
            if remainder != 0:
                padding_num = self.sp_degree - remainder
                x = torch.cat(
                    [x, x.new_zeros(B, padding_num, D)],
                    dim=1
                )

        if is_enable_sequence_parallel():
            x = split(x, 1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            num_modalities=2,   # 告诉 SelfAttention / RoPE 是双模态
        )

        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, **kwargs)

        # head：对 2S 个 token 一起预测 patch
        x = self.head(x, e)  # [B, 2S(+pad), P*out_dim]

        if is_enable_sequence_parallel():
            x = gather(x, 1)

        if remainder != 0:
            x = x[:, :-padding_num]

        # 如果 use_fixed_seq_len 且 pad 了，这里剪掉 padding
        if self.use_fixed_seq_len and tokens_num_total < x.shape[1]:
            x = x[:, :tokens_num_total]

        # 4) 拆回 video / flow token 序列
        x_v = x[:, :tokens_per_modality]           # [B, S, P*out_dim]
        x_f = x[:, tokens_per_modality:2*tokens_per_modality]

        # 5) 分别 unpatchify 成 [B, out_dim, T, H, W]
        video = self.unpatchify(x_v, tt, th, tw)
        flow_out = self.unpatchify(x_f, tt, th, tw)

        # 裁掉 padding
        video = video[:, :, :, :ori_height, :ori_width]
        flow_out = flow_out[:, :, :, :ori_height, :ori_width]

        # 返回两个模态
        return video, flow_out
    
    def unpatchify(self, x, t, h, w):
        c = self.out_dim
        pt, ph, pw = self.patch_size
        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = torch.einsum("nfhwpqrc->ncfphqwr", x)
        out = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return out
    
    def init_weights(self):
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        
        # init output layer
        nn.init.zeros_(self.head.head.weight)


@MODELS.register_module("WanX21")
def wanx_21_t2v(from_pretrained=None, **kwargs):
    # 单模态版本：保持原来的行为
    kwargs.setdefault("num_modalities", 1)
    model = Transformer(**kwargs)
    # if from_pretrained is not None:
    #     print(f"load wanx model from {from_pretrained}")
    #     load_checkpoint(model, from_pretrained)
    # else:
    #     print(f"wanx from_pretrained is None, init wanx model by random")
    print(f"init wanx model by random")
    return model


@MODELS.register_module("WanX21VideoJAM")
def wanx_21_t2v_videojam(from_pretrained=None, base_latent_channels=16, **kwargs):
    """
    双模态 VideoJAM 版：
    - in_dim = out_dim = base_latent_channels （每个模态的通道数）
    - num_modalities = 2（video + flow）
    - forward(x_video, t, context, seq_len, flow=x_flow)
        -> 返回 (video_pred, flow_pred)
    """
    kwargs.setdefault("in_dim", base_latent_channels)
    kwargs.setdefault("out_dim", base_latent_channels)
    kwargs.setdefault("num_modalities", 2)

    model = Transformer(**kwargs)

    if from_pretrained is not None:
        print(f"[VideoJAM] load base WanX model from {from_pretrained}")
        load_checkpoint(model, from_pretrained)
    else:
        print("[VideoJAM] from_pretrained is None, use random init as base")

    print(
        f"[VideoJAM] Dual-modality Transformer: in_dim={model.in_dim}, "
        f"out_dim={model.out_dim}, num_modalities={model.num_modalities}"
    )

    return model
