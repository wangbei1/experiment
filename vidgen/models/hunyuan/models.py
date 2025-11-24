from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from .attenion import attention, get_cu_seqlens
from .posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate
from .token_refiner import SingleTokenRefiner
from vidgen.acceleration.checkpoint import auto_grad_checkpoint
from vidgen.registry import MODELS
from vidgen.utils.ckpt_utils import load_checkpoint
from transformers import PretrainedConfig, PreTrainedModel

from training_acc.dist.parallel_state import is_enable_sequence_parallel
from training_acc.patches.hunyuan import split, gather, collect_tokens, collect_heads
from vidgen.utils.train_utils import unmask_tokens, mask_rope_embedding_hunyuan


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6,
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias,
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6,)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6,)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias,
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6,
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6,
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias,
        )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6,)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6,)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias,
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6,
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
        )

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)
                
        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)
        
        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            
        if is_enable_sequence_parallel():
            img_q = collect_tokens(img_q)
            img_k = collect_tokens(img_k)
            img_v = collect_tokens(img_v)

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)
        
        if is_enable_sequence_parallel():
            txt_q = split(txt_q, 2)
            txt_k = split(txt_k, 2)
            txt_v = split(txt_v, 2)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        attn = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=img_k.shape[0],
        )
        
        img_attn, txt_attn = attn[:, :-txt_len], attn[:, -txt_len:]
        
        if is_enable_sequence_parallel():
            img_attn = collect_heads(img_attn, self.heads_num)
            txt_attn = gather(txt_attn, 2)

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(
            hidden_size, hidden_size * 3 + mlp_hidden_dim,
        )
        # proj and mlp_out
        self.linear2 = nn.Linear(
            hidden_size + mlp_hidden_dim, hidden_size,
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6,)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6,)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6,
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
        )

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)            

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)
            
        if is_enable_sequence_parallel():
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]
            
            img_q = collect_tokens(img_q)
            img_k = collect_tokens(img_k)
            img_v = collect_tokens(img_v)
            
            txt_q = split(txt_q, 2)
            txt_k = split(txt_k, 2)
            txt_v = split(txt_v, 2)
            
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)
            v = torch.cat((img_v, txt_v), dim=1)

        # Compute attention.
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        attn = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=x.shape[0],
        )
        
        if is_enable_sequence_parallel():
            img_attn, txt_attn = attn[:, :-txt_len], attn[:, -txt_len:]
            
            img_attn = collect_heads(img_attn, self.heads_num)            
            txt_attn = gather(txt_attn, 2) 
            attn = torch.cat((img_attn, txt_attn), dim=1)

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate)

class HYVideoDiffusionTransformerConfig(PretrainedConfig):
    """
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    """
    model_type = "HYVideoDiffusionTransformer"
    def __init__(
        self,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        patch_size: list = [1, 2, 2],
        in_channels: int = 16,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        maskdit: bool = False,
        decoder_depth: int = 0,
    ):
        
        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.mlp_width_ratio = mlp_width_ratio
        self.mlp_act_type = mlp_act_type
        self.mm_double_blocks_depth = mm_double_blocks_depth
        self.mm_single_blocks_depth = mm_single_blocks_depth
        self.rope_dim_list = rope_dim_list
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.qk_norm_type = qk_norm_type
        self.guidance_embed = guidance_embed
        self.text_projection = text_projection
        self.use_attention_mask = use_attention_mask
        self.maskdit = maskdit
        self.decoder_depth = decoder_depth
        
        super().__init__()

class HYVideoDiffusionTransformer(PreTrainedModel):
    config_class = HYVideoDiffusionTransformerConfig
    def __init__(self, config):
        super().__init__(config)

        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels if config.out_channels is None else config.out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = config.guidance_embed
        self.rope_dim_list = config.rope_dim_list

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = config.use_attention_mask
        self.text_projection = config.text_projection

        self.text_states_dim = config.text_states_dim
        self.text_states_dim_2 = config.text_states_dim_2

        if config.hidden_size % config.heads_num != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = config.hidden_size // config.heads_num
        if sum(self.rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {self.rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = config.hidden_size
        self.heads_num = config.heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size,
        )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, self.hidden_size, self.heads_num, depth=2,
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu")
        )

        # text modulation
        self.vector_in = MLPEmbedder(
            self.text_states_dim_2, self.hidden_size
        )

        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu")
            )
            if config.guidance_embed
            else None
        )

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=config.mlp_width_ratio,
                    mlp_act_type=config.mlp_act_type,
                    qk_norm=config.qk_norm,
                    qk_norm_type=config.qk_norm_type,
                    qkv_bias=config.qkv_bias,
                )
                for _ in range(config.mm_double_blocks_depth)
            ]
        )

        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=config.mlp_width_ratio,
                    mlp_act_type=config.mlp_act_type,
                    qk_norm=config.qk_norm,
                    qk_norm_type=config.qk_norm_type,
                )
                for _ in range(config.mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
        )
        
        if config.maskdit:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_size))
            nn.init.normal_(self.mask_token, std=.02)
        self.encoder_depth = config.mm_single_blocks_depth - config.decoder_depth if config.maskdit else config.mm_single_blocks_depth

    def forward(
        self,
        img: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        xt_mask=None,
        xs_mask=None,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = {}
        # img = x
        txt = text_states
        
        T, ori_height, ori_width = img.shape[-3:]
        if ori_width % self.config.patch_size[2] != 0:
            img = F.pad(img, (0, self.config.patch_size[2] - ori_width % self.config.patch_size[2]))
        if ori_height % self.config.patch_size[1] != 0:
            img = F.pad(img, (0, 0, 0, self.config.patch_size[1] - ori_height % self.config.patch_size[1]))
        
        _, _, ot, oh, ow = img.shape
        
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(self.rope_dim_list, [tt, th, tw], theta=256, use_real=True, theta_rescale_factor=1)
        
        
        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                guidance = torch.ones_like(t) * 6 * 1000
            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        
        # mask tokens
        if xs_mask is not None:
            B, T, S = xs_mask["ids_keep"].shape
            S_orig = th*tw
            # mask hidden states
            img = rearrange(img, "B (T S) C -> B T S C", T=T, S=S_orig)
            img = torch.gather(img, dim=2, index=xs_mask["ids_keep"].unsqueeze(-1).repeat(1, 1, 1, img.shape[-1]))
            img = rearrange(img, "B T S C -> B (T S) C", T=T, S=S)
            # mask rope embedding
            freqs_cos_orig, freqs_sin_orig = freqs_cos.clone(), freqs_sin.clone()
            freqs_cos = mask_rope_embedding_hunyuan(freqs_cos, xs_mask["ids_keep"], B, T)
            freqs_sin = mask_rope_embedding_hunyuan(freqs_sin, xs_mask["ids_keep"], B, T)
            
            
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q
        
        if is_enable_sequence_parallel():
            img = split(img, 1)
            rope_split_dim = 0 if xs_mask is None else 1 #测试一下
            freqs_cos = split(freqs_cos, rope_split_dim)
            freqs_sin = split(freqs_sin, rope_split_dim)
            
            
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        for _, block in enumerate(self.double_blocks):
            img, txt = auto_grad_checkpoint(block, img, txt, vec, txt_seq_len, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, freqs_cis)

        # Merge txt and img to pass through single stream blocks.
        img = torch.cat((img, txt), 1)
        if len(self.single_blocks) > 0:
            for _, block in enumerate(self.single_blocks[:self.encoder_depth]):
                img = auto_grad_checkpoint(block, img, vec, txt_seq_len, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, freqs_cis)

        if xs_mask is not None:
            img_x = img[:, :-txt_seq_len, ...]
            
            if is_enable_sequence_parallel():
                img_x = gather(img_x, 1)
                
            img_x = unmask_tokens(img_x, xs_mask["ids_restore"], self.mask_token)
            
            # Recompute cu_squlens and max_seqlen for flash attention
            img_seq_len = img_x.shape[1]
            cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
            cu_seqlens_kv = cu_seqlens_q
            max_seqlen_q = img_seq_len + txt_seq_len
            max_seqlen_kv = max_seqlen_q
            
            if is_enable_sequence_parallel():
                img_x = split(img_x, 1)
                freqs_cos_orig = split(freqs_cos_orig, 0)
                freqs_sin_orig = split(freqs_sin_orig, 0)
                        
            img = torch.cat((img_x, img[:, -txt_seq_len:, ...]), 1)
            
            if len(self.single_blocks) > 0:
                freqs_cis_orig = (freqs_cos_orig, freqs_sin_orig) if freqs_cos_orig is not None else None
                for _, block in enumerate(self.single_blocks[self.encoder_depth:]):
                    img = auto_grad_checkpoint(block, img, vec, txt_seq_len, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, freqs_cis_orig)
                    
        img = img[:, :-txt_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        
        if is_enable_sequence_parallel():
            img = gather(img, 1)

        img = self.unpatchify(img, tt, th, tw)
        img = img[:, :, :, :ori_height, :ori_width]
        return img

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts
    
@MODELS.register_module("Hunyuan")
def hunyuan_dit(from_pretrained=None, **kwargs):
    force_huggingface = kwargs.pop("force_huggingface", False)
    if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
        model = HYVideoDiffusionTransformer.from_pretrained(from_pretrained, **kwargs)
    else:
        config = HYVideoDiffusionTransformerConfig(**kwargs)
        model = HYVideoDiffusionTransformer(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model