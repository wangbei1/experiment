# import os
# import torch
# import torch.nn as nn
# import torch.amp as amp
# import math
# import torch.nn.functional as F
# from .attention import flash_attention
# import torch.distributed as dist
# from vidgen.registry import MODELS
# from vidgen.utils.ckpt_utils import load_checkpoint
# from vidgen.acceleration.checkpoint import auto_grad_checkpoint

# from training_acc.dist.parallel_state import is_enable_sequence_parallel
# from training_acc.patches.wanx2_1_t2v import split, gather, collect_tokens, collect_heads


# __all__ = ['Transformer']

# def sinusoidal_embedding_1d(dim, position):
#     # preprocess
#     assert dim % 2 == 0
#     half = dim // 2
#     position = position.type(torch.float64)

#     # calculation
#     sinusoid = torch.outer(
#         position,
#         torch.pow(10000, -torch.arange(half).to(position).div(half))
#     )
#     x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
#     return x


# @amp.autocast("cuda", enabled=False)
# def rope_params(max_seq_len, dim, theta=10000):
#     assert dim % 2 == 0
#     freqs = torch.outer(
#         torch.arange(max_seq_len),
#         1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
#     )
#     freqs = torch.polar(torch.ones_like(freqs), freqs)
#     return freqs


# @amp.autocast("cuda", enabled=False)
# def rope_apply(x, grid_sizes, freqs):
#     n, c = x.size(2), x.size(3) // 2

#     # split freqs
#     freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

#     # loop over samples
#     output = []
#     for i, (f, h, w) in enumerate(grid_sizes.tolist()):
#         seq_len = f * h * w

#         # precompute multipliers
#         x_i = torch.view_as_complex(
#             x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
#         )
#         freqs_i = torch.cat([
#             freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
#             freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
#             freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
#         ], dim=-1).reshape(seq_len, 1, -1)

#         # apply rotary embedding
#         x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
#         x_i = torch.cat([x_i, x[i, seq_len:]])

#         # append to collection
#         output.append(x_i)
#     return torch.stack(output).to(x.dtype)


# class RMSNorm(nn.Module):

#     def __init__(self, dim, eps=1e-5):
#         super().__init__()
#         self.dim = dim
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))
    
#     def forward(self, x):
#         return self._norm(x.float()).type_as(x) * self.weight
    
#     def _norm(self, x):
#         return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


# class LayerNorm(nn.LayerNorm):
#     def __init__(self, dim, eps=1e-6, elementwise_affine=False):
#         super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)
    
#     def forward(self, x):
#         return super().forward(x)


# class SelfAttention(nn.Module):

#     def __init__(
#         self,
#         dim,
#         num_heads,
#         window_size=(-1, -1),
#         qk_norm=True,
#         eps=1e-6
#     ):
#         assert dim % num_heads == 0
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.window_size = window_size
#         self.qk_norm = qk_norm
#         self.eps = eps

#         # layers
#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(dim, dim)
#         self.v = nn.Linear(dim, dim)
#         self.o = nn.Linear(dim, dim)
#         self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
#         self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
#     def forward(self, x, seq_lens, grid_sizes, freqs):
#         b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

#         # query, key, value function
#         def qkv_fn(x):
#             q = self.norm_q(self.q(x)).view(b, s, n, d)
#             k = self.norm_k(self.k(x)).view(b, s, n, d)
#             v = self.v(x).view(b, s, n, d)
#             return q, k, v
        
#         q, k, v = qkv_fn(x)

#         if is_enable_sequence_parallel():
#             q = collect_tokens(q)
#             k = collect_tokens(k)
#             v = collect_tokens(v)

#         x = flash_attention(
#             q=rope_apply(q, grid_sizes, freqs),
#             k=rope_apply(k, grid_sizes, freqs),
#             v=v,
#             k_lens=seq_lens,
#             window_size=self.window_size
#         )

#         if is_enable_sequence_parallel():
#             x = collect_heads(x)

#         # output
#         x = x.flatten(2)
#         x = self.o(x)
#         return x


# class CrossAttention(SelfAttention):

#     def forward(self, x, context, context_lens):
#         """
#         x:              [B, L1, C].
#         context:        [B, L2, C].
#         context_lens:   [B].
#         """
#         b, n, d = x.size(0), self.num_heads, self.head_dim

#         # compute query, key, value
#         q = self.norm_q(self.q(x)).view(b, -1, n, d)
#         k = self.norm_k(self.k(context)).view(b, -1, n, d)
#         v = self.v(context).view(b, -1, n, d)

#         if is_enable_sequence_parallel():
#             q = collect_tokens(q)
#             k = split(k, 2)
#             v = split(v, 2)

#         # compute attention
#         x = flash_attention(q, k, v, k_lens=context_lens)

#         if is_enable_sequence_parallel():
#             x = collect_heads(x)

#         # output
#         x = x.flatten(2)
#         x = self.o(x)
#         return x


# class AttentionBlock(nn.Module):
    
#     def __init__(
#         self,
#         dim,
#         ffn_dim,
#         num_heads,
#         window_size=(-1, -1),
#         qk_norm=True,
#         cross_attn_norm=False,
#         eps=1e-6
#     ):
#         super().__init__()
#         self.dim = dim
#         self.ffn_dim = ffn_dim
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.qk_norm = qk_norm
#         self.cross_attn_norm = cross_attn_norm
#         self.eps = eps

#         # layers
#         self.norm1 = LayerNorm(dim, eps)
#         self.self_attn = SelfAttention(dim, num_heads, window_size, qk_norm, eps)
#         self.norm3 = LayerNorm(
#             dim, eps, elementwise_affine=True
#         ) if cross_attn_norm else nn.Identity()
#         self.cross_attn = CrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
#         self.norm2 = LayerNorm(dim, eps)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, ffn_dim),
#             nn.GELU(approximate='tanh'),
#             nn.Linear(ffn_dim, dim)
#         )

#         # modulation
#         self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
    
#     # 没有设置环境变量或者enable是false的时候，disable是true，即不启用
#     @torch.compile(disable=os.getenv('ENABLE_COMPILE', 'FALSE').lower() == 'false')
#     def forward(
#         self,
#         x,
#         e,
#         seq_lens,
#         grid_sizes,
#         freqs,
#         context,
#         context_lens,
#     ):
#         e = (self.modulation + e).chunk(6, dim=1)

#         # self-attention
#         y = self.self_attn(
#             self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs
#         )
#         x = x + y * e[2]

#         # cross-attention & ffn function
#         def cross_attn_ffn(x, context, context_lens, e):
#             x = x + self.cross_attn(self.norm3(x), context, context_lens)
#             y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
#             x = x + y * e[5]
#             return x

#         x = cross_attn_ffn(x, context, context_lens, e)
#         return x


# class Head(nn.Module):

#     def __init__(self, dim, out_dim, patch_size, eps=1e-6):
#         super().__init__()
#         self.dim = dim
#         self.out_dim = out_dim
#         self.patch_size = patch_size
#         self.eps = eps

#         # layers
#         out_dim = math.prod(patch_size) * out_dim
#         self.norm = LayerNorm(dim, eps)
#         self.head = nn.Linear(dim, out_dim)

#         # modulation
#         self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)
    
#     def forward(self, x, e):
#         e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
#         x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
#         return x


# class Transformer(nn.Module):

#     def __init__(
#         self,
#         patch_size=(1, 2, 2),
#         text_len=512,
#         in_dim=16,
#         dim=2048,
#         ffn_dim=8192,
#         freq_dim=256,
#         text_dim=4096,
#         out_dim=16,
#         num_heads=16,
#         num_layers=32,
#         window_size=(-1, -1),
#         qk_norm=True,
#         cross_attn_norm=False,
#         eps=1e-6,
#         use_fixed_seq_len=True,
#         sp_degree=1,
#         use_dual_head: bool = False,   # <<< 新增
#     ):
#         super().__init__()
#         self.patch_size = patch_size
#         self.text_len = text_len
#         self.in_dim = in_dim
#         self.dim = dim
#         self.ffn_dim = ffn_dim
#         self.freq_dim = freq_dim
#         self.text_dim = text_dim
#         self.out_dim = out_dim            # 注意：VideoJAM 时这里是 32
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#         self.window_size = window_size
#         self.qk_norm = qk_norm
#         self.cross_attn_norm = cross_attn_norm
#         self.eps = eps
#         self.use_fixed_seq_len = use_fixed_seq_len
#         self.sp_degree = sp_degree
#         self.use_dual_head = use_dual_head   # <<< 新增

#         # embeddings
#         self.patch_embedding = nn.Conv3d(
#             in_dim, dim, kernel_size=patch_size, stride=patch_size
#         )
#         self.text_embedding = nn.Sequential(
#             nn.Linear(text_dim, dim),
#             nn.GELU(approximate='tanh'),
#             nn.Linear(dim, dim)
#         )

#         self.time_embedding = nn.Sequential(
#             nn.Linear(freq_dim, dim),
#             nn.SiLU(),
#             nn.Linear(dim, dim)
#         )
#         self.time_projection = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(dim, dim * 6)
#         )

#         # blocks
#         self.blocks = nn.ModuleList([AttentionBlock(
#             dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps
#         ) for _ in range(num_layers)])

#         # ===== head: 单 head or 双 head =====
#         if self.use_dual_head:
#             # 这里约定 out_dim = video + flow 总通道数（例如 32）
#             assert out_dim % 2 == 0, "dual head 目前默认均分通道数"
#             self.out_dim_video = out_dim // 2
#             self.out_dim_flow  = out_dim - self.out_dim_video

#             self.head_video = Head(dim, self.out_dim_video, patch_size, eps)
#             self.head_flow  = Head(dim, self.out_dim_flow,  patch_size, eps)
#         else:
#             # 原始 WanX：单 head
#             self.head = Head(dim, out_dim, patch_size, eps)

#         # buffers ...
#         assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
#         self._freqs_initialized = False



#     def expand_to_videojam(self, base_channels: int = 16):
#         """
#         正确扩展 WanX21 Transformer 到 VideoJAM 版（双通道：视频+flow）

#         要求：
#         - 原始 in_dim = out_dim = base_channels (16)
#         - 扩展后 in_dim = out_dim = 2 * base_channels (32)
#         - 当输入为 [video_latent, zeros] 时：
#                 前 16 通道的视频输出 与 原模型 完全一致
#                 后 16 通道（flow）输出 = 0
#         """
#         if self.in_dim == base_channels * 2 and self.out_dim == base_channels * 2:
#             print("[VideoJAM] already expanded, skip.")
#             return

#         assert self.in_dim == base_channels, f"in_dim={self.in_dim}, expect {base_channels}"
#         assert self.out_dim == base_channels, f"out_dim={self.out_dim}, expect {base_channels}"

#         new_in_dim  = base_channels * 2
#         new_out_dim = base_channels * 2

#         # ===== 1) patch_embedding: 扩输入通道 =====
#         pe = self.patch_embedding
#         with torch.no_grad():
#             w = pe.weight    # [D, C_old, pt, ph, pw]
#             D, C_old, pt, ph, pw = w.shape
#             assert C_old == base_channels
#             C_new = new_in_dim

#             w_new = torch.zeros(D, C_new, pt, ph, pw,
#                                 device=w.device, dtype=w.dtype)
#             # 前 16 通道复制原权重
#             w_new[:, :C_old, :, :, :] = w

#             pe.weight = nn.Parameter(w_new)

#             # bias 一般是 [D]，和通道数无关，不需要改
#             # 如果你实现里 bias 有通道维，这里再单独处理

#         self.in_dim = new_in_dim

#         # ===== 2) head: 正确扩展输出通道（关键部分） =====
#         head = self.head
#         P = head.patch_size[0] * head.patch_size[1] * head.patch_size[2]  # =4

#         with torch.no_grad():
#             # 原 head.linear
#             old_linear: nn.Linear = head.head
#             F_old, D_in = old_linear.weight.shape   # [P*C_old, dim]
#             assert F_old == P * base_channels

#             b_old = old_linear.bias                 # [P*C_old]

#             # 先 reshape 成 [P, C_old, dim]
#             W_old = old_linear.weight.view(P, base_channels, D_in)
#             b_old_2d = b_old.view(P, base_channels)  # [P, C_old]

#             # 新的 [P, C_new, dim]，后 16 通道为 0
#             C_new = new_out_dim
#             W_new = torch.zeros(P, C_new, D_in,
#                                 device=W_old.device, dtype=W_old.dtype)
#             b_new_2d = torch.zeros(P, C_new,
#                                 device=b_old_2d.device, dtype=b_old_2d.dtype)

#             # 保证每个 patch 位置上，前 16 通道的权重和 bias 一模一样
#             W_new[:, :base_channels, :] = W_old
#             b_new_2d[:, :base_channels] = b_old_2d

#             # 再展回 [P*C_new, dim]
#             W_new_flat = W_new.view(P * C_new, D_in)
#             b_new_flat = b_new_2d.view(P * C_new)

#             # 构造新的 Linear
#             new_linear = nn.Linear(D_in, P * C_new, bias=True)
#             new_linear.weight = nn.Parameter(W_new_flat)
#             new_linear.bias   = nn.Parameter(b_new_flat)

#             head.head = new_linear
#             head.out_dim = new_out_dim
#             self.out_dim = new_out_dim

#         print(f"[VideoJAM] Expanded Transformer: in_dim {base_channels} -> {new_in_dim}, "
#             f"out_dim {base_channels} -> {new_out_dim}")

#     @property
#     def freqs(self):
#         if not self._freqs_initialized or not hasattr(self, '_freqs'):
#             # 动态初始化保障逻辑
#             d = self.dim // self.num_heads
#             device = self.patch_embedding.weight.device
            
#             # 确保在CUDA设备生成张量
#             with torch.cuda.device(device):
#                 self._freqs = torch.cat([
#                     rope_params(1024, d-4*(d//6)),
#                     rope_params(1024, 2*(d//6)),
#                     rope_params(1024, 2*(d//6))
#                 ], dim=1).to(device)
                
#             self._freqs_initialized = True
#         return self._freqs
    
#     def forward(
#         self,
#         x,
#         t,
#         context,
#         seq_len
#     ):
#         """
#         x:              A list of videos each with shape [C, T, H, W].
#         t:              [B].
#         context:        A list of text embeddings each with shape [L, C].
#         """
#         # params
#         device = self.patch_embedding.weight.device
#         if self.freqs.device != device:
#             self.freqs = self.freqs.to(device)
                
#         T, ori_height, ori_width = x.shape[-3:]
#         if ori_width % self.patch_size[2] != 0:
#             x = F.pad(x, (0, self.patch_size[2] - ori_width % self.patch_size[2]))
#         if ori_height % self.patch_size[1] != 0:
#             x = F.pad(x, (0, 0, 0, self.patch_size[1] - ori_height % self.patch_size[1]))
        
#         _, _, ot, oh, ow = x.shape
         
#         tt, th, tw = (
#             ot // self.patch_size[0],
#             oh // self.patch_size[1],
#             ow // self.patch_size[2],
#         )
        
#         # embeddings
#         x = self.patch_embedding(x)
#         grid_sizes = torch.stack([
#             torch.tensor(u.shape[1:], dtype=torch.long) for u in x
#         ])
#         x = x.flatten(2).transpose(1, 2)
#         seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)

#         tokens_num = x.shape[1]
#         remainder = 0
#         if self.use_fixed_seq_len:
#             # use fixed seq length
#             assert tokens_num <= seq_len, f"{seq_len=}, {x[0].shape=}"
#             padding_num = seq_len - tokens_num
#             x = torch.cat([x, x.new_zeros(x.shape[0], padding_num, x.shape[2])], dim=1)
#         else:
#             remainder = tokens_num % int(self.sp_degree) #兼容sp
#             if remainder != 0:
#                 padding_num = self.sp_degree - remainder
#                 x = torch.cat([x, x.new_zeros(x.shape[0], padding_num, x.shape[2])], dim=1)
        
#         # time embeddings
#         e = self.time_embedding(
#             sinusoidal_embedding_1d(self.freq_dim, t).to(x.dtype)
#         )
#         e0 = self.time_projection(e).unflatten(1, (6, self.dim))

#         # context
#         context_lens = None
#         context = self.text_embedding(torch.stack([torch.cat([
#             u, u.new_zeros(self.text_len - u.size(0), u.size(1))
#         ]) for u in context]))

#         if is_enable_sequence_parallel():
#             x = split(x, 1)

#         # arguments
#         kwargs = dict(
#             e=e0,
#             seq_lens=seq_lens,
#             grid_sizes=grid_sizes,
#             freqs=self.freqs,
#             context=context,
#             context_lens=context_lens
#         )

#         for block in self.blocks:
#             x = auto_grad_checkpoint(block, x, **kwargs)
        
#         # head
#         x = self.head(x, e)

#         if is_enable_sequence_parallel():
#             x = gather(x, 1)

#         if remainder != 0:
#             x = x[:, :-padding_num]
            
#         # unpatchify
#         x = self.unpatchify(x, tt, th, tw)
        
#         x = x[:, :, :, :ori_height, :ori_width]
        
#         return x
    
#     def unpatchify(self, x, t, h, w):
#         c = self.out_dim
#         pt, ph, pw = self.patch_size
#         x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
#         x = torch.einsum("nfhwpqrc->ncfphqwr", x)
#         out = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        
#         return out
    
#     def init_weights(self):
#         # basic init
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
        
#         # init embeddings
#         nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
#         for m in self.text_embedding.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=.02)
#         for m in self.time_embedding.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=.02)
        
#         # init output layer
#         nn.init.zeros_(self.head.head.weight)


# @MODELS.register_module("WanX21")
# def wanx_21_t2v(from_pretrained=None, **kwargs):
#     model = Transformer(**kwargs)
#     # if from_pretrained is not None:
#     #     print(f"load wanx model from {from_pretrained}")
#     #     load_checkpoint(model, from_pretrained)
#     # else:
#     #     print(f"wanx from_pretrained is None, init wanx model by random")
#     print(f"init wanx model by random")
#     return model

# @MODELS.register_module("WanX21VideoJAM")
# def wanx_21_t2v_videojam(from_pretrained=None, base_latent_channels=16, **kwargs):
#     """
#     - 先按普通 WanX21 构建 + load_pretrained（16 通道）
#     - 再扩成 VideoJAM 双输入/双输出（32 通道）
#     """
#     model = Transformer(**kwargs)

#     # 1. 正常加载 16 通道的 WanX 预训练权重
#     if from_pretrained is not None:
#         print(f"[VideoJAM] load base WanX model from {from_pretrained}")
#         load_checkpoint(model, from_pretrained)
#     else:
#         print("[VideoJAM] from_pretrained is None, use random init as base")

#     # 2. 扩成 video+flow 的 2C 通道
#     model.expand_to_videojam(base_channels=base_latent_channels)
#     print(
#         f"[VideoJAM] Expanded Transformer: in_dim {base_latent_channels} -> {model.in_dim}, "
#         f"out_dim {base_latent_channels} -> {model.out_dim}"
#     )

#     return model


# import os
# import torch
# import torch.nn as nn
# import torch.amp as amp
# import math
# import torch.nn.functional as F
# from .attention import flash_attention
# import torch.distributed as dist
# from vidgen.registry import MODELS
# from vidgen.utils.ckpt_utils import load_checkpoint
# from vidgen.acceleration.checkpoint import auto_grad_checkpoint

# from training_acc.dist.parallel_state import is_enable_sequence_parallel
# from training_acc.patches.wanx2_1_t2v import split, gather, collect_tokens, collect_heads


# __all__ = ['Transformer']


# def sinusoidal_embedding_1d(dim, position):
#     # preprocess
#     assert dim % 2 == 0
#     half = dim // 2
#     position = position.type(torch.float64)

#     # calculation
#     sinusoid = torch.outer(
#         position,
#         torch.pow(10000, -torch.arange(half).to(position).div(half))
#     )
#     x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
#     return x


# @amp.autocast("cuda", enabled=False)
# def rope_params(max_seq_len, dim, theta=10000):
#     assert dim % 2 == 0
#     freqs = torch.outer(
#         torch.arange(max_seq_len),
#         1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
#     )
#     freqs = torch.polar(torch.ones_like(freqs), freqs)
#     return freqs


# @amp.autocast("cuda", enabled=False)
# def rope_apply(x, grid_sizes, freqs, num_modalities: int = 1):
#     """
#     x:              [B, L, n_heads, head_dim]
#     grid_sizes:     [B, 3] 每个样本的 (T', H', W')
#     freqs:          [max_seq, head_dim/2] 预先生成的频率
#     num_modalities: 1 = 单模态 (原逻辑)
#                     2 = 双模态，假设序列是 [video_tokens, flow_tokens]，
#                         两段长度相同，且共享同一套 (t,h,w) RoPE，相位一致
#     """
#     n, c = x.size(2), x.size(3) // 2  # c = head_dim / 2 (复数维)

#     # split freqs 为 T / H / W 三段
#     freqs_t, freqs_h, freqs_w = freqs.split(
#         [c - 2 * (c // 3), c // 3, c // 3], dim=1
#     )

#     output = []
#     B, L = x.shape[0], x.shape[1]

#     for i, (f, h, w) in enumerate(grid_sizes.tolist()):
#         seq_len_per = f * h * w                 # 每个模态的 token 数
#         total_len = seq_len_per * num_modalities  # video+flow 的总 token 数
#         total_len = min(total_len, L)           # 防御一下越界

#         # 只对前 total_len 个 token 做 RoPE，后面的（padding）不变
#         x_main = x[i, :total_len].to(torch.float64)  # [total_len, n, 2c]
#         x_complex = torch.view_as_complex(
#             x_main.reshape(total_len, n, -1, 2)
#         )  # [total_len, n, c]

#         # 构造一套基础 (t,h,w) 的频率表 [seq_len_per, 1, c]
#         base_freqs = torch.cat([
#             freqs_t[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
#             freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
#             freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1)
#         ], dim=-1).reshape(seq_len_per, 1, -1)  # [seq_len_per, 1, c]

#         if num_modalities == 1:
#             freqs_i = base_freqs  # [seq_len_per, 1, c]
#         else:
#             # 把同一套 (t,h,w) 相位复制给每个模态：
#             # [seq_len_per * num_modalities, 1, c]
#             freqs_i = base_freqs.repeat(num_modalities, 1, 1)

#         freqs_i = freqs_i[:total_len]  # 防御一下

#         # 应用 RoPE：复数乘法 = 二维旋转
#         x_rot = torch.view_as_real(x_complex * freqs_i).flatten(2)  # [total_len, n, 2c]

#         # 把没处理的 padding token 拼回去
#         if total_len < L:
#             x_rest = x[i, total_len:]
#             x_out = torch.cat([x_rot, x_rest], dim=0)
#         else:
#             x_out = x_rot

#         output.append(x_out)

#     return torch.stack(output).to(x.dtype)


# class RMSNorm(nn.Module):

#     def __init__(self, dim, eps=1e-5):
#         super().__init__()
#         self.dim = dim
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(dim))
    
#     def forward(self, x):
#         return self._norm(x.float()).type_as(x) * self.weight
    
#     def _norm(self, x):
#         return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


# class LayerNorm(nn.LayerNorm):
#     def __init__(self, dim, eps=1e-6, elementwise_affine=False):
#         super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)
    
#     def forward(self, x):
#         return super().forward(x)


# class SelfAttention(nn.Module):

#     def __init__(
#         self,
#         dim,
#         num_heads,
#         window_size=(-1, -1),
#         qk_norm=True,
#         eps=1e-6
#     ):
#         assert dim % num_heads == 0
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.window_size = window_size
#         self.qk_norm = qk_norm
#         self.eps = eps

#         # layers
#         self.q = nn.Linear(dim, dim)
#         self.k = nn.Linear(dim, dim)
#         self.v = nn.Linear(dim, dim)
#         self.o = nn.Linear(dim, dim)
#         self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
#         self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
#     def forward(self, x, seq_lens, grid_sizes, freqs, num_modalities: int = 1):
#         b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

#         # query, key, value function
#         def qkv_fn(x):
#             q = self.norm_q(self.q(x)).view(b, s, n, d)
#             k = self.norm_k(self.k(x)).view(b, s, n, d)
#             v = self.v(x).view(b, s, n, d)
#             return q, k, v
        
#         q, k, v = qkv_fn(x)

#         if is_enable_sequence_parallel():
#             q = collect_tokens(q)
#             k = collect_tokens(k)
#             v = collect_tokens(v)

#         x = flash_attention(
#             q=rope_apply(q, grid_sizes, freqs, num_modalities=num_modalities),
#             k=rope_apply(k, grid_sizes, freqs, num_modalities=num_modalities),
#             v=v,
#             k_lens=seq_lens,
#             window_size=self.window_size
#         )

#         if is_enable_sequence_parallel():
#             x = collect_heads(x)

#         # output
#         x = x.flatten(2)
#         x = self.o(x)
#         return x


# class CrossAttention(SelfAttention):

#     def forward(self, x, context, context_lens):
#         """
#         x:              [B, L1, C].
#         context:        [B, L2, C].
#         context_lens:   [B].
#         """
#         b, n, d = x.size(0), self.num_heads, self.head_dim

#         # compute query, key, value
#         q = self.norm_q(self.q(x)).view(b, -1, n, d)
#         k = self.norm_k(self.k(context)).view(b, -1, n, d)
#         v = self.v(context).view(b, -1, n, d)

#         if is_enable_sequence_parallel():
#             q = collect_tokens(q)
#             k = split(k, 2)
#             v = split(v, 2)

#         # compute attention
#         x = flash_attention(q, k, v, k_lens=context_lens)

#         if is_enable_sequence_parallel():
#             x = collect_heads(x)

#         # output
#         x = x.flatten(2)
#         x = self.o(x)
#         return x


# class AttentionBlock(nn.Module):
    
#     def __init__(
#         self,
#         dim,
#         ffn_dim,
#         num_heads,
#         window_size=(-1, -1),
#         qk_norm=True,
#         cross_attn_norm=False,
#         eps=1e-6
#     ):
#         super().__init__()
#         self.dim = dim
#         self.ffn_dim = ffn_dim
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.qk_norm = qk_norm
#         self.cross_attn_norm = cross_attn_norm
#         self.eps = eps

#         # layers
#         self.norm1 = LayerNorm(dim, eps)
#         self.self_attn = SelfAttention(dim, num_heads, window_size, qk_norm, eps)
#         self.norm3 = LayerNorm(
#             dim, eps, elementwise_affine=True
#         ) if cross_attn_norm else nn.Identity()
#         self.cross_attn = CrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
#         self.norm2 = LayerNorm(dim, eps)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, ffn_dim),
#             nn.GELU(approximate='tanh'),
#             nn.Linear(ffn_dim, dim)
#         )

#         # modulation
#         self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
    
#     # 没有设置环境变量或者enable是false的时候，disable是true，即不启用
#     @torch.compile(disable=os.getenv('ENABLE_COMPILE', 'FALSE').lower() == 'false')
#     def forward(
#         self,
#         x,
#         e,
#         seq_lens,
#         grid_sizes,
#         freqs,
#         context,
#         context_lens,
#         num_modalities: int = 1,
#     ):
#         e = (self.modulation + e).chunk(6, dim=1)

#         # self-attention
#         y = self.self_attn(
#             self.norm1(x) * (1 + e[1]) + e[0],
#             seq_lens,
#             grid_sizes,
#             freqs,
#             num_modalities=num_modalities
#         )
#         x = x + y * e[2]

#         # cross-attention & ffn function
#         def cross_attn_ffn(x, context, context_lens, e):
#             x = x + self.cross_attn(self.norm3(x), context, context_lens)
#             y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
#             x = x + y * e[5]
#             return x

#         x = cross_attn_ffn(x, context, context_lens, e)
#         return x


# class Head(nn.Module):

#     def __init__(self, dim, out_dim, patch_size, eps=1e-6):
#         super().__init__()
#         self.dim = dim
#         self.out_dim = out_dim
#         self.patch_size = patch_size
#         self.eps = eps

#         # layers
#         out_dim_proj = math.prod(patch_size) * out_dim
#         self.norm = LayerNorm(dim, eps)
#         self.head = nn.Linear(dim, out_dim_proj)

#         # modulation
#         self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)
    
#     def forward(self, x, e):
#         e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
#         x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
#         return x


# class Transformer(nn.Module):

#     def __init__(
#         self,
#         patch_size=(1, 2, 2),
#         text_len=512,
#         in_dim=16,
#         dim=2048,
#         ffn_dim=8192,
#         freq_dim=256,
#         text_dim=4096,
#         out_dim=16,
#         num_heads=16,
#         num_layers=32,
#         window_size=(-1, -1),
#         qk_norm=True,
#         cross_attn_norm=False,
#         eps=1e-6,
#         use_fixed_seq_len=True,
#         sp_degree=1,
#         num_modalities: int = 2,      # 1: 原始单模态；2: video+flow 双模态
#     ):
#         super().__init__()
#         self.patch_size = patch_size
#         self.text_len = text_len
#         self.in_dim = in_dim
#         self.dim = dim
#         self.ffn_dim = ffn_dim
#         self.freq_dim = freq_dim
#         self.text_dim = text_dim
#         self.out_dim = out_dim
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#         self.window_size = window_size
#         self.qk_norm = qk_norm
#         self.cross_attn_norm = cross_attn_norm
#         self.eps = eps
#         self.use_fixed_seq_len = use_fixed_seq_len
#         self.sp_degree = sp_degree
#         self.num_modalities = num_modalities

#         # embeddings
#         self.patch_embedding = nn.Conv3d(
#             in_dim, dim, kernel_size=patch_size, stride=patch_size
#         )
#         self.text_embedding = nn.Sequential(
#             nn.Linear(text_dim, dim),
#             nn.GELU(approximate='tanh'),
#             nn.Linear(dim, dim)
#         )

#         self.time_embedding = nn.Sequential(
#             nn.Linear(freq_dim, dim),
#             nn.SiLU(),
#             nn.Linear(dim, dim)
#         )
#         self.time_projection = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(dim, dim * 6)
#         )

#         # modality embeddings（用于区分 video / flow）
#         if self.num_modalities > 1:
#             self.mod_embeddings = nn.Parameter(
#                 torch.zeros(self.num_modalities, dim)
#             )
#         else:
#             self.mod_embeddings = None

#         # blocks
#         self.blocks = nn.ModuleList([AttentionBlock(
#             dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps
#         ) for _ in range(num_layers)])

#         # head（每个模态输出 out_dim 通道）
#         self.head = Head(dim, out_dim, patch_size, eps)

#         # buffers (don't use register_buffer otherwise dtype will be changed in to())
#         assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
#         self._freqs_initialized = False

#     # 原来的 expand_to_videojam 可以保留（不再强依赖），这里就不动了
#     def expand_to_videojam(self, base_channels: int = 16):
#         ...

#     @property
#     def freqs(self):
#         if not self._freqs_initialized or not hasattr(self, '_freqs'):
#             # 动态初始化保障逻辑
#             d = self.dim // self.num_heads
#             device = self.patch_embedding.weight.device
            
#             # 确保在CUDA设备生成张量
#             with torch.cuda.device(device):
#                 self._freqs = torch.cat([
#                     rope_params(1024, d-4*(d//6)),
#                     rope_params(1024, 2*(d//6)),
#                     rope_params(1024, 2*(d//6))
#                 ], dim=1).to(device)
                
#             self._freqs_initialized = True
#         return self._freqs
    
#     def forward(
#         self,
#         x,
#         t,
#         context,
#         seq_len,
#         flow=None,          # 如果提供 flow，则走双模态分支
#     ):
#         """
#         单模态:
#             x:        [B, C, T, H, W]
#         双模态:
#             x:        [B, C, T, H, W]  (video)
#             flow:     [B, C, T, H, W]  (flow)
#         t:            [B]
#         context:      一个 list / tensor，每个元素 [L, C_text]
#         seq_len:      最大 token 长度（通常 = T'*H'*W' 或 2*T'*H'*W'）
#         """
#         device = self.patch_embedding.weight.device
#         if self.freqs.device != device:
#             self._freqs = self.freqs.to(device)

#         # -------- 统一 padding（对 video / flow 一起） --------
#         T, ori_height, ori_width = x.shape[-3:]
#         pad_w = (self.patch_size[2] - ori_width % self.patch_size[2]) % self.patch_size[2]
#         pad_h = (self.patch_size[1] - ori_height % self.patch_size[1]) % self.patch_size[1]

#         if pad_w != 0 or pad_h != 0:
#             x = F.pad(x, (0, pad_w, 0, pad_h))
#             if flow is not None:
#                 flow = F.pad(flow, (0, pad_w, 0, pad_h))

#         _, _, ot, oh, ow = x.shape
#         tt, th, tw = (
#             ot // self.patch_size[0],
#             oh // self.patch_size[1],
#             ow // self.patch_size[2],
#         )

#         # -------- time embeddings --------
#         e = self.time_embedding(
#             sinusoidal_embedding_1d(self.freq_dim, t).to(x.dtype)
#         )  # [B, dim]
#         e0 = self.time_projection(e).unflatten(1, (6, self.dim))  # [B, 6, dim]

#         # -------- context --------
#         context_lens = None
#         context = self.text_embedding(torch.stack([torch.cat([
#             u, u.new_zeros(self.text_len - u.size(0), u.size(1))
#         ]) for u in context]))

#         # ========== 单模态路径：保持原有行为 ==========
#         if self.num_modalities == 1 or flow is None:
#             # embeddings
#             x = self.patch_embedding(x)  # [B, D, T', H', W']
#             grid_sizes = torch.stack([
#                 torch.tensor(u.shape[1:], dtype=torch.long) for u in x
#             ])
#             x = x.flatten(2).transpose(1, 2)  # [B, S, D]
#             seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)

#             tokens_num = x.shape[1]
#             remainder = 0
#             if self.use_fixed_seq_len:
#                 # use fixed seq length
#                 assert tokens_num <= seq_len, f"{seq_len=}, {x[0].shape=}"
#                 padding_num = seq_len - tokens_num
#                 if padding_num > 0:
#                     x = torch.cat(
#                         [x, x.new_zeros(x.shape[0], padding_num, x.shape[2])],
#                         dim=1
#                     )
#             else:
#                 remainder = tokens_num % int(self.sp_degree)  # 兼容sp
#                 if remainder != 0:
#                     padding_num = self.sp_degree - remainder
#                     x = torch.cat(
#                         [x, x.new_zeros(x.shape[0], padding_num, x.shape[2])],
#                         dim=1
#                     )

#             if is_enable_sequence_parallel():
#                 x = split(x, 1)

#             kwargs = dict(
#                 e=e0,
#                 seq_lens=seq_lens,
#                 grid_sizes=grid_sizes,
#                 freqs=self.freqs,
#                 context=context,
#                 context_lens=context_lens,
#                 num_modalities=1,
#             )

#             for block in self.blocks:
#                 x = auto_grad_checkpoint(block, x, **kwargs)
            
#             # head
#             x = self.head(x, e)

#             if is_enable_sequence_parallel():
#                 x = gather(x, 1)

#             if remainder != 0:
#                 x = x[:, :-padding_num]
                
#             # unpatchify
#             x = self.unpatchify(x, tt, th, tw)
#             x = x[:, :, :, :ori_height, :ori_width]
#             return x

#         # ========== 双模态路径：video + flow ==========
#         # 1) patch embedding 分别处理 video / flow
#         x_v = self.patch_embedding(x)         # [B, D, T', H', W']
#         x_f = self.patch_embedding(flow)      # [B, D, T', H', W']

#         grid_sizes = torch.stack([
#             torch.tensor(u.shape[1:], dtype=torch.long) for u in x_v
#         ])  # 每个样本一行 [T', H', W']

#         x_v = x_v.flatten(2).transpose(1, 2)  # [B, S, D]
#         x_f = x_f.flatten(2).transpose(1, 2)  # [B, S, D]

#         B, S, D = x_v.shape
#         tokens_per_modality = S

#         # 2) 加模态 embedding（区分 video / flow）
#         if self.mod_embeddings is not None:
#             x_v = x_v + self.mod_embeddings[0].view(1, 1, -1)
#             x_f = x_f + self.mod_embeddings[1].view(1, 1, -1)

#         # 3) concat 成一个长序列：[video_tokens, flow_tokens]
#         x = torch.cat([x_v, x_f], dim=1)      # [B, 2S, D]
#         tokens_num_total = x.shape[1]

#         seq_lens = torch.full(
#             (B,),
#             tokens_num_total,
#             dtype=torch.long
#         )

#         remainder = 0
#         if self.use_fixed_seq_len:
#             assert tokens_num_total <= seq_len, f"{seq_len=}, {tokens_num_total=}"
#             padding_num = seq_len - tokens_num_total
#             if padding_num > 0:
#                 x = torch.cat(
#                     [x, x.new_zeros(B, padding_num, D)],
#                     dim=1
#                 )
#         else:
#             remainder = tokens_num_total % int(self.sp_degree)
#             if remainder != 0:
#                 padding_num = self.sp_degree - remainder
#                 x = torch.cat(
#                     [x, x.new_zeros(B, padding_num, D)],
#                     dim=1
#                 )

#         if is_enable_sequence_parallel():
#             x = split(x, 1)

#         kwargs = dict(
#             e=e0,
#             seq_lens=seq_lens,
#             grid_sizes=grid_sizes,
#             freqs=self.freqs,
#             context=context,
#             context_lens=context_lens,
#             num_modalities=2,   # 告诉 SelfAttention / RoPE 是双模态
#         )

#         for block in self.blocks:
#             x = auto_grad_checkpoint(block, x, **kwargs)

#         # head：对 2S 个 token 一起预测 patch
#         x = self.head(x, e)  # [B, 2S(+pad), P*out_dim]

#         if is_enable_sequence_parallel():
#             x = gather(x, 1)

#         if remainder != 0:
#             x = x[:, :-padding_num]

#         # 如果 use_fixed_seq_len 且 pad 了，这里剪掉 padding
#         if self.use_fixed_seq_len and tokens_num_total < x.shape[1]:
#             x = x[:, :tokens_num_total]

#         # 4) 拆回 video / flow token 序列
#         x_v = x[:, :tokens_per_modality]           # [B, S, P*out_dim]
#         x_f = x[:, tokens_per_modality:2*tokens_per_modality]

#         # 5) 分别 unpatchify 成 [B, out_dim, T, H, W]
#         video = self.unpatchify(x_v, tt, th, tw)
#         flow_out = self.unpatchify(x_f, tt, th, tw)

#         # 裁掉 padding
#         video = video[:, :, :, :ori_height, :ori_width]
#         flow_out = flow_out[:, :, :, :ori_height, :ori_width]

#         # 返回两个模态
#         return video, flow_out
    
#     def unpatchify(self, x, t, h, w):
#         c = self.out_dim
#         pt, ph, pw = self.patch_size
#         x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
#         x = torch.einsum("nfhwpqrc->ncfphqwr", x)
#         out = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
#         return out
    
#     def init_weights(self):
#         # basic init
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
        
#         # init embeddings
#         nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
#         for m in self.text_embedding.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=.02)
#         for m in self.time_embedding.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, std=.02)
        
#         # init output layer
#         nn.init.zeros_(self.head.head.weight)


# @MODELS.register_module("WanX21")
# def wanx_21_t2v(from_pretrained=None, **kwargs):
#     # 单模态版本：保持原来的行为
#     kwargs.setdefault("num_modalities", 2)
#     model = Transformer(**kwargs)
#     # if from_pretrained is not None:
#     #     print(f"load wanx model from {from_pretrained}")
#     #     load_checkpoint(model, from_pretrained)
#     # else:
#     #     print(f"wanx from_pretrained is None, init wanx model by random")
#     print(f"init wanx model by random")
#     return model


# @MODELS.register_module("WanX21VideoJAM")
# def wanx_21_t2v_videojam(from_pretrained=None, base_latent_channels=16, **kwargs):
#     """
#     双模态 VideoJAM 版：
#     - in_dim = out_dim = base_latent_channels （每个模态的通道数）
#     - num_modalities = 2（video + flow）
#     - forward(x_video, t, context, seq_len, flow=x_flow)
#         -> 返回 (video_pred, flow_pred)
#     """
#     kwargs.setdefault("in_dim", base_latent_channels)
#     kwargs.setdefault("out_dim", base_latent_channels)
#     kwargs.setdefault("num_modalities", 2)

#     model = Transformer(**kwargs)

#     if from_pretrained is not None:
#         print(f"[VideoJAM] load base WanX model from {from_pretrained}")
#         load_checkpoint(model, from_pretrained)
#     else:
#         print("[VideoJAM] from_pretrained is None, use random init as base")

#     print(
#         f"[VideoJAM] Dual-modality Transformer: in_dim={model.in_dim}, "
#         f"out_dim={model.out_dim}, num_modalities={model.num_modalities}"
#     )

#     return model

import os
import math
import torch
import torch.nn as nn
import torch.amp as amp
import torch.nn.functional as F

from .attention import flash_attention
import torch.distributed as dist
from vidgen.registry import MODELS
from vidgen.utils.ckpt_utils import load_checkpoint
from vidgen.acceleration.checkpoint import auto_grad_checkpoint

from training_acc.dist.parallel_state import is_enable_sequence_parallel
from training_acc.patches.wanx2_1_t2v import split, gather, collect_tokens, collect_heads

__all__ = ["Transformer"]


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    sinusoid = torch.outer(
        position,
        torch.pow(10000, -torch.arange(half).to(position).div(half)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0
        / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
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
        eps=1e-6,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x_):
            q_ = self.norm_q(self.q(x_)).view(b, s, n, d)
            k_ = self.norm_k(self.k(x_)).view(b, s, n, d)
            v_ = self.v(x_).view(b, s, n, d)
            return q_, k_, v_

        q, k, v = qkv_fn(x)

        if is_enable_sequence_parallel():
            q = collect_tokens(q)
            k = collect_tokens(k)
            v = collect_tokens(v)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        if is_enable_sequence_parallel():
            x = collect_heads(x)

        x = x.flatten(2)
        x = self.o(x)
        return x


class CrossAttention(SelfAttention):
    def forward(self, x, context, context_lens):
        """
        x:        [B, L1, C]
        context:  [B, L2, C]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        if is_enable_sequence_parallel():
            q = collect_tokens(q)
            k = split(k, 2)
            v = split(v, 2)

        x = flash_attention(q, k, v, k_lens=context_lens)

        if is_enable_sequence_parallel():
            x = collect_heads(x)

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
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = LayerNorm(dim, eps)
        self.self_attn = SelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = CrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = LayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    @torch.compile(disable=os.getenv("ENABLE_COMPILE", "FALSE").lower() == "false")
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        e = (self.modulation + e).chunk(6, dim=1)

        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0],
            seq_lens,
            grid_sizes,
            freqs,
        )
        x = x + y * e[2]

        def cross_attn_ffn(x_, context_, context_lens_, e_):
            x_ = x_ + self.cross_attn(self.norm3(x_), context_, context_lens_)
            y_ = self.ffn(self.norm2(x_) * (1 + e_[4]) + e_[3])
            x_ = x_ + y_ * e_[5]
            return x_

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim_total = math.prod(patch_size) * out_dim
        self.norm = LayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim_total)

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
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
        use_dual_head: bool = False,  # ⭐ 双 head 开关
    ):
        super().__init__()
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim  # 总通道数（VideoJAM 时 32）
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_fixed_seq_len = use_fixed_seq_len
        self.sp_degree = sp_degree
        self.use_dual_head = use_dual_head

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # ⭐ 单 head / 双 head
        if self.use_dual_head:
            assert (
                out_dim % 2 == 0
            ), "use_dual_head=True 时 out_dim 必须是偶数（例如 32）"
            self.out_dim_video = out_dim // 2
            self.out_dim_flow = out_dim - self.out_dim_video
            self.head_video = Head(dim, self.out_dim_video, patch_size, eps)
            self.head_flow = Head(dim, self.out_dim_flow, patch_size, eps)
        else:
            self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self._freqs_initialized = False
        self._freqs = None

    # ================= VideoJAM 扩展 =================
    def expand_to_videojam(self, base_channels: int = 16):
        """
        从 16ch WanX 扩成 32ch VideoJAM 版（双输入 + 双 head）.

        - in_dim: 16 -> 32
        - patch_embedding: 输入通道扩一倍，前 16 复制，后 16 置零
        - head: 原 head -> head_video，新建 head_flow
        - 最终 out_dim = 32，对外输出 [video(16), flow(16)]
        """
        if self.in_dim == base_channels * 2 and getattr(self, "use_dual_head", False):
            print("[VideoJAM] already expanded to dual-head version, skip.")
            return

        assert self.in_dim == base_channels, f"in_dim={self.in_dim}, expect {base_channels}"
        assert self.out_dim == base_channels, f"out_dim={self.out_dim}, expect {base_channels}"
        assert not getattr(self, "use_dual_head", False), \
            "expand_to_videojam 只在单 head 的 16ch 模型上调用"

        new_in_dim = base_channels * 2
        new_out_dim = base_channels * 2

        # 1) patch_embedding: 扩输入通道（前一半复制，后一半 0）
        old_pe = self.patch_embedding
        with torch.no_grad():
            w = old_pe.weight  # [D, C_old, pt, ph, pw]
            D, C_old, pt, ph, pw = w.shape
            assert C_old == base_channels
            C_new = new_in_dim

            w_new = torch.zeros(D, C_new, pt, ph, pw,
                                device=w.device, dtype=w.dtype)
            w_new[:, :C_old, :, :, :] = w  # 前 16 通道复制

            new_pe = nn.Conv3d(
                in_channels=C_new,
                out_channels=D,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=old_pe.bias is not None,
            )
            new_pe.weight = nn.Parameter(w_new)
            if old_pe.bias is not None:
                new_pe.bias = nn.Parameter(old_pe.bias.data.clone())

            self.patch_embedding = new_pe

        self.in_dim = new_in_dim

        # 2) head: 原 head -> video head，新建 flow head
        self.head_video = self.head
        self.out_dim_video = base_channels

        self.head_flow = Head(self.dim, base_channels, self.patch_size, self.eps)
        with torch.no_grad():
            vh_w = self.head_video.head.weight
            std = vh_w.std().item()
            nn.init.normal_(self.head_flow.head.weight, mean=0.0, std=std)
            if self.head_flow.head.bias is not None:
                nn.init.zeros_(self.head_flow.head.bias)

        self.out_dim_flow = base_channels
        self.out_dim = new_out_dim
        self.use_dual_head = True

        # 删除原 alias，避免 safetensors 共享内存报错
        del self.head

        print(
            f"[VideoJAM] Expanded Transformer to dual-head: "
            f"in_dim {base_channels} -> {new_in_dim}, "
            f"out_dim {base_channels}x2, "
            f"video head from pretrained, flow head random-init."
        )

    # ================= RoPE ================
    @property
    def freqs(self):
        if (not self._freqs_initialized) or (self._freqs is None):
            d = self.dim // self.num_heads
            device = self.patch_embedding.weight.device
            self._freqs = torch.cat(
                [
                    rope_params(1024, d - 4 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                ],
                dim=1,
            ).to(device)
            self._freqs_initialized = True
        return self._freqs

    def forward(self, x, t, context, seq_len):
        device = self.patch_embedding.weight.device
        _ = self.freqs
        if self._freqs.device != device:
            self._freqs = self._freqs.to(device)

        T, ori_height, ori_width = x.shape[-3:]
        if ori_width % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - ori_width % self.patch_size[2]))
        if ori_height % self.patch_size[1] != 0:
            x = F.pad(
                x,
                (0, 0, 0, self.patch_size[1] - ori_height % self.patch_size[1]),
            )

        _, _, ot, oh, ow = x.shape

        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        x = self.patch_embedding(x)
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[1:], dtype=torch.long, device=x.device) for u in x]
        )
        x = x.flatten(2).transpose(1, 2)
        seq_lens = torch.tensor(
            [u.size(0) for u in x],
            dtype=torch.long,
            device=x.device,
        )

        tokens_num = x.shape[1]
        remainder = 0
        if self.use_fixed_seq_len:
            assert tokens_num <= seq_len, f"{seq_len=}, {x[0].shape=}"
            padding_num = seq_len - tokens_num
            x = torch.cat(
                [x, x.new_zeros(x.shape[0], padding_num, x.shape[2])],
                dim=1,
            )
        else:
            remainder = tokens_num % int(self.sp_degree)
            if remainder != 0:
                padding_num = self.sp_degree - remainder
                x = torch.cat(
                    [x, x.new_zeros(x.shape[0], padding_num, x.shape[2])],
                    dim=1,
                )

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(x.dtype).to(x.device)
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        context_lens = None
        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))]
                    )
                    for u in context
                ]
            )
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
        )

        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, **kwargs)

        # ===== head & unpatchify =====
        if self.use_dual_head:
            x_video = self.head_video(x, e)  # [B, tokens, P*C_v]
            x_flow = self.head_flow(x, e)    # [B, tokens, P*C_f]

            if is_enable_sequence_parallel():
                x_video = gather(x_video, 1)
                x_flow = gather(x_flow, 1)

            if remainder != 0:
                x_video = x_video[:, :-padding_num]
                x_flow = x_flow[:, :-padding_num]

            x_video = self._unpatchify_with_c(
                x_video, tt, th, tw, self.out_dim_video
            )
            x_flow = self._unpatchify_with_c(
                x_flow, tt, th, tw, self.out_dim_flow
            )

            x = torch.cat([x_video, x_flow], dim=1)
        else:
            x = self.head(x, e)
            if is_enable_sequence_parallel():
                x = gather(x, 1)
            if remainder != 0:
                x = x[:, :-padding_num]
            x = self.unpatchify(x, tt, th, tw)

        x = x[:, :, :, :ori_height, :ori_width]
        return x

    def _unpatchify_with_c(self, x, t, h, w, c):
        pt, ph, pw = self.patch_size
        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = torch.einsum("nfhwpqrc->ncfphqwr", x)
        out = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return out

    def unpatchify(self, x, t, h, w):
        return self._unpatchify_with_c(x, t, h, w, self.out_dim)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        if hasattr(self, "head"):
            nn.init.zeros_(self.head.head.weight)
        if hasattr(self, "head_video"):
            nn.init.zeros_(self.head_video.head.weight)
        if hasattr(self, "head_flow"):
            nn.init.zeros_(self.head_flow.head.weight)


@MODELS.register_module("WanX21")
def wanx_21_t2v(from_pretrained=None, **kwargs):
    model = Transformer(**kwargs)
    print("init wanx model by random")
    return model
