import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_varlen_qkvpacked_func
from torch.nn.attention import sdpa_kernel

# import .context_parallel as cp
from .layers import (
    FeedForward,
    PatchEmbed,
    RMSNorm,
    TimestepEmbedder,
)
from .mod_rmsnorm import modulated_rmsnorm
from .residual_tanh_gated_rmsnorm import (
    residual_tanh_gated_rmsnorm,
)
from .rope_mixed import (
    compute_mixed_rotation,
    create_position_matrix,
)
from .temporal_rope import apply_rotary_emb_qk_real
from .utils import (
    AttentionPool,
    modulate,
    pad_and_split_xy,
    unify_streams,
)
from vidgen.acceleration.checkpoint import auto_grad_checkpoint
from vidgen.registry import MODELS
from vidgen.utils.ckpt_utils import load_checkpoint
from transformers import PretrainedConfig, PreTrainedModel


class AsymmetricAttention(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        update_y: bool = True,
        out_bias: bool = True,
        softmax_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_heads = num_heads
        self.head_dim = dim_x // num_heads
        self.update_y = update_y
        self.softmax_scale = softmax_scale
        if dim_x % num_heads != 0:
            raise ValueError(
                f"dim_x={dim_x} should be divisible by num_heads={num_heads}"
            )

        # Input layers.
        self.qkv_bias = qkv_bias
        self.qkv_x = nn.Linear(dim_x, 3 * dim_x, bias=qkv_bias, device=device)
        # Project text features to match visual features (dim_y -> dim_x)
        self.qkv_y = nn.Linear(dim_y, 3 * dim_x, bias=qkv_bias, device=device)

        # Query and key normalization for stability.
        assert qk_norm
        self.q_norm_x = RMSNorm(self.head_dim, device=device)
        self.k_norm_x = RMSNorm(self.head_dim, device=device)
        self.q_norm_y = RMSNorm(self.head_dim, device=device)
        self.k_norm_y = RMSNorm(self.head_dim, device=device)

        # Output layers. y features go back down from dim_x -> dim_y.
        self.proj_x = nn.Linear(dim_x, dim_x, bias=out_bias, device=device)
        self.proj_y = (
            nn.Linear(dim_x, dim_y, bias=out_bias, device=device)
            if update_y
            else nn.Identity()
        )

    def prepare_qkv(
        self,
        x: torch.Tensor,  # (B, N, dim_x)
        y: torch.Tensor,  # (B, L, dim_y)
        *,
        scale_x: torch.Tensor,
        scale_y: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ):
        # Pre-norm for visual features
        x = modulated_rmsnorm(x, scale_x)  # (B, M, dim_x) where M = N / cp_group_size

        # Process visual features
        qkv_x = self.qkv_x(x)  # (B, M, 3 * dim_x)
        assert qkv_x.dtype == torch.bfloat16
        qkv_x = qkv_x.view(qkv_x.size(0), qkv_x.size(1), 3, self.num_heads, self.head_dim)

        # Split qkv_x into q, k, v
        q_x, k_x, v_x = qkv_x.unbind(2)  # (B, N, local_h, head_dim)
        q_x = self.q_norm_x(q_x)
        q_x = apply_rotary_emb_qk_real(q_x, rope_cos, rope_sin)
        k_x = self.k_norm_x(k_x)
        k_x = apply_rotary_emb_qk_real(k_x, rope_cos, rope_sin)
        
        # Process text features
        y = modulated_rmsnorm(y, scale_y)  # (B, L, dim_y)
        qkv_y = self.qkv_y(y)  # (B, L, 3 * dim)
        qkv_y = qkv_y.view(qkv_y.size(0), qkv_y.size(1), 3, self.num_heads, self.head_dim)
        q_y, k_y, v_y = qkv_y.unbind(2)
        q_y = self.q_norm_y(q_y)
        k_y = self.k_norm_y(k_y)

        q = torch.cat([q_x, q_y], dim=1).permute(0, 2, 1, 3)
        k = torch.cat([k_x, k_y], dim=1).permute(0, 2, 1, 3)
        v = torch.cat([v_x, v_y], dim=1).permute(0, 2, 1, 3)
        return q, k, v

    def run_attention(
        self,
        q: torch.Tensor,  # B, h, (N + L), c
        k: torch.Tensor,  # B, h, (N + L), c
        v: torch.Tensor,  # B, h, (N + L), c
        B: int,
        L: int,
        N: int,
        attn_mask: torch.Tensor,
    ):
        local_dim = self.num_heads * self.head_dim
        
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=self.softmax_scale
        ) #B, h, (N + L), c
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        x, y = out.split([N, L], dim=1)
        
        assert x.size() == (B, N, local_dim)
        assert y.size() == (B, L, local_dim)

        x = self.proj_x(x)  # (B, M, dim_x)

        y = self.proj_y(y)  # (B, L, dim_y)
        return x, y

    def forward(
        self,
        x: torch.Tensor,  # (B, N, dim_x)
        y: torch.Tensor,  # (B, L, dim_y)
        *,
        scale_x: torch.Tensor,  # (B, dim_x), modulation for pre-RMSNorm.
        scale_y: torch.Tensor,  # (B, dim_y), modulation for pre-RMSNorm.
        attn_mask: Dict[str, torch.Tensor] = None,
        **rope_rotation,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of asymmetric multi-modal attention.

        Args:
            x: (B, N, dim_x) tensor for visual tokens
            y: (B, L, dim_y) tensor of text token features
            packed_indices: Dict with keys for Flash Attention
            num_frames: Number of frames in the video. N = num_frames * num_spatial_tokens

        Returns:
            x: (B, N, dim_x) tensor of visual tokens after multi-modal attention
            y: (B, L, dim_y) tensor of text token features after multi-modal attention
        """
        B, L, _ = y.shape
        _, N, _ = x.shape

        # Predict a packed QKV tensor from visual and text features.
        # Don't checkpoint the all_to_all.
        q, k, v = self.prepare_qkv(
            x=x,
            y=y,
            scale_x=scale_x,
            scale_y=scale_y,
            rope_cos=rope_rotation.get("rope_cos"),
            rope_sin=rope_rotation.get("rope_sin"),
        )  # (total <= B * (N + L), 3, local_heads, head_dim)

        x, y = self.run_attention(
            q, k, v,
            B=B,
            L=L,
            N=N,
            attn_mask=attn_mask,
        )
        return x, y

class AsymmetricJointBlock(nn.Module):
    def __init__(
        self,
        hidden_size_x: int,
        hidden_size_y: int,
        num_heads: int,
        *,
        mlp_ratio_x: float = 8.0,  # Ratio of hidden size to d_model for MLP for visual tokens.
        mlp_ratio_y: float = 4.0,  # Ratio of hidden size to d_model for MLP for text tokens.
        update_y: bool = True,  # Whether to update text tokens in this block.
        device: Optional[torch.device] = None,
        **block_kwargs,
    ):
        super().__init__()
        self.update_y = update_y
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.mod_x = nn.Linear(hidden_size_x, 4 * hidden_size_x, device=device)
        if self.update_y:
            self.mod_y = nn.Linear(hidden_size_x, 4 * hidden_size_y, device=device)
        else:
            self.mod_y = nn.Linear(hidden_size_x, hidden_size_y, device=device)

        # Self-attention:
        self.attn = AsymmetricAttention(
            hidden_size_x,
            hidden_size_y,
            num_heads=num_heads,
            update_y=update_y,
            device=device,
            **block_kwargs,
        )

        # MLP.
        mlp_hidden_dim_x = int(hidden_size_x * mlp_ratio_x)
        assert mlp_hidden_dim_x == int(1536 * 8)
        self.mlp_x = FeedForward(
            in_features=hidden_size_x,
            hidden_size=mlp_hidden_dim_x,
            multiple_of=256,
            ffn_dim_multiplier=None,
            device=device,
        )

        # MLP for text not needed in last block.
        if self.update_y:
            mlp_hidden_dim_y = int(hidden_size_y * mlp_ratio_y)
            self.mlp_y = FeedForward(
                in_features=hidden_size_y,
                hidden_size=mlp_hidden_dim_y,
                multiple_of=256,
                ffn_dim_multiplier=None,
                device=device,
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        y: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor,
    ):
        """Forward pass of a block.

        Args:
            x: (B, N, dim) tensor of visual tokens
            c: (B, dim) tensor of conditioned features
            y: (B, L, dim) tensor of text tokens
            num_frames: Number of frames in the video. N = num_frames * num_spatial_tokens

        Returns:
            x: (B, N, dim) tensor of visual tokens after block
            y: (B, L, dim) tensor of text tokens after block
        """
        N = x.size(1)

        c = F.silu(c)
        mod_x = self.mod_x(c)
        scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = mod_x.chunk(4, dim=1)

        mod_y = self.mod_y(c)
        if self.update_y:
            scale_msa_y, gate_msa_y, scale_mlp_y, gate_mlp_y = mod_y.chunk(4, dim=1)
        else:
            scale_msa_y = mod_y

        # Self-attention block.
        x_attn, y_attn = self.attn(
            x,
            y,
            scale_x=scale_msa_x,
            scale_y=scale_msa_y,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            attn_mask=attn_mask,
        )

        assert x_attn.size(1) == N
        x = residual_tanh_gated_rmsnorm(x, x_attn, gate_msa_x)
        if self.update_y:
            y = residual_tanh_gated_rmsnorm(y, y_attn, gate_msa_y)

        # MLP block.
        x = self.ff_block_x(x, scale_mlp_x, gate_mlp_x)
        if self.update_y:
            y = self.ff_block_y(y, scale_mlp_y, gate_mlp_y)

        return x, y

    def ff_block_x(self, x, scale_x, gate_x):
        x_mod = modulated_rmsnorm(x, scale_x)
        x_res = self.mlp_x(x_mod)
        x = residual_tanh_gated_rmsnorm(x, x_res, gate_x)  # Sandwich norm
        return x

    def ff_block_y(self, y, scale_y, gate_y):
        y_mod = modulated_rmsnorm(y, scale_y)
        y_res = self.mlp_y(y_mod)
        y = residual_tanh_gated_rmsnorm(y, y_res, gate_y)  # Sandwich norm
        return y


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, device=device
        )
        self.mod = nn.Linear(hidden_size, 2 * hidden_size, device=device)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, device=device
        )

    def forward(self, x, c):
        c = F.silu(c)
        shift, scale = self.mod(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class AsymmDiTJointConfig(PretrainedConfig):
    model_type = "AsymmDiTJoint"

    def __init__(
        self,
        patch_size=2,
        in_channels=12,
        hidden_size_x=3072,
        hidden_size_y=1536,
        depth=48,
        num_heads=24,
        mlp_ratio_x=4.0,
        mlp_ratio_y=4.0,
        t5_feat_dim: int = 4096,
        t5_token_length: int = 256,
        patch_embed_bias: bool = True,
        timestep_mlp_bias: bool = True,
        timestep_scale: Optional[float] = None,
        use_extended_posenc: bool = False,
        rope_theta: float = 10000.0,
        device: Optional[torch.device] = None,
        **block_kwargs
    ):
        self.depth = depth
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y
        self.mlp_ratio_x = mlp_ratio_x
        self.mlp_ratio_y = mlp_ratio_y
        self.in_channels = in_channels
        self.patch_embed_bias = patch_embed_bias
        self.timestep_mlp_bias = timestep_mlp_bias
        self.timestep_scale = timestep_scale
        self.t5_feat_dim = t5_feat_dim
        self.t5_token_length = t5_token_length
        self.rope_theta = rope_theta
        self.use_extended_posenc = use_extended_posenc
        self.device = device
        self.block_kwargs = block_kwargs
        super().__init__()
        
class AsymmDiTJoint(PreTrainedModel):
    """
    Diffusion model with a Transformer backbone.

    Ingests text embeddings instead of a label.
    """
    config_class = AsymmDiTJointConfig

    def __init__(self, config):
        super().__init__(config)
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.patch_size = config.patch_size
        self.num_heads = config.num_heads
        self.hidden_size_x = config.hidden_size_x
        self.hidden_size_y = config.hidden_size_y
        self.head_dim = (
            self.hidden_size_x // self.num_heads
        )  # Head dimension and count is determined by visual.
        self.use_extended_posenc = config.use_extended_posenc
        self.t5_token_length = config.t5_token_length
        self.t5_feat_dim = config.t5_feat_dim
        self.rope_theta = (
            config.rope_theta  # Scaling factor for frequency computation for temporal RoPE.
        )

        self.x_embedder = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.hidden_size_x,
            bias=config.patch_embed_bias,
            device=config.device,
        )
        # Conditionings
        # Timestep
        self.t_embedder = TimestepEmbedder(
            self.hidden_size_x, bias=config.timestep_mlp_bias, timestep_scale=config.timestep_scale
        )

        # Caption Pooling (T5)
        self.t5_y_embedder = AttentionPool(
            self.t5_feat_dim, num_heads=8, output_dim=self.hidden_size_x, device=config.device
        )

        # Dense Embedding Projection (T5)
        self.t5_yproj = nn.Linear(
            self.t5_feat_dim, self.hidden_size_y, bias=True, device=config.device
        )

        # Initialize pos_frequencies as an empty parameter.
        self.pos_frequencies = nn.Parameter(
            torch.empty(3, self.num_heads, self.head_dim // 2, device=config.device)
        )

        # for depth 48:
        #  b =  0: AsymmetricJointBlock, update_y=True
        #  b =  1: AsymmetricJointBlock, update_y=True
        #  ...
        #  b = 46: AsymmetricJointBlock, update_y=True
        #  b = 47: AsymmetricJointBlock, update_y=False. No need to update text features.
        blocks = []
        for b in range(config.depth):
            # Joint multi-modal block
            update_y = b < config.depth - 1
            block = AsymmetricJointBlock(
                self.hidden_size_x,
                self.hidden_size_y,
                self.num_heads,
                mlp_ratio_x=config.mlp_ratio_x,
                mlp_ratio_y=config.mlp_ratio_y,
                update_y=update_y,
                device=config.device,
                **config.block_kwargs,
            )

            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.final_layer = FinalLayer(
            self.hidden_size_x, self.patch_size, self.out_channels, device=config.device
        )

    def embed_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C=12, T, H, W) tensor of visual tokens

        Returns:
            x: (B, C=3072, N) tensor of visual tokens with positional embedding.
        """
        return self.x_embedder(x)  # Convert BcTHW to BCN

    def prepare(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        t5_feat: torch.Tensor,
        t5_mask: torch.Tensor,
    ):
        """Prepare input and conditioning embeddings."""
        # Visual patch embeddings with positional encoding.
        T, H, W = x.shape[-3:]
        pH, pW = H // self.patch_size, W // self.patch_size
        x = self.embed_x(x)  # (B, N, D), where N = T * H * W / patch_size ** 2
        assert x.ndim == 3
        B = x.size(0)

        # Construct position array of size [N, 3].
        # pos[:, 0] is the frame index for each location,
        # pos[:, 1] is the row index for each location, and
        # pos[:, 2] is the column index for each location.
        pH, pW = H // self.patch_size, W // self.patch_size
        N = T * pH * pW
        assert x.size(1) == N
        pos = create_position_matrix(
            T, pH=pH, pW=pW, device=x.device, dtype=torch.float32
        )  # (N, 3)
        rope_cos, rope_sin = compute_mixed_rotation(
            freqs=self.pos_frequencies, pos=pos
        )  # Each are (N, num_heads, dim // 2)

        # Global vector embedding for conditionings.
        c_t = self.t_embedder(1 - sigma)  # (B, D)

        # Pool T5 tokens using attention pooler
        # Note y_feat[1] contains T5 token features.
        assert (
            t5_feat.size(1) == self.t5_token_length
        ), f"Expected L={self.t5_token_length}, got {t5_feat.shape} for y_feat."
        t5_y_pool = self.t5_y_embedder(t5_feat, t5_mask)  # (B, D)
        assert (
            t5_y_pool.size(0) == B
        ), f"Expected B={B}, got {t5_y_pool.shape} for t5_y_pool."

        c = c_t + t5_y_pool

        y_feat = self.t5_yproj(t5_feat)  # (B, L, t5_feat_dim) --> (B, L, D)

        return x, c, y_feat, rope_cos, rope_sin

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        y_feat: List[torch.Tensor],
        y_mask: List[torch.Tensor],
        packed_indices: Dict[str, torch.Tensor] = None,
        rope_cos: torch.Tensor = None,
        rope_sin: torch.Tensor = None,
        **kwargs
    ):
        """Forward pass of DiT.

        Args:
            x: (B, C, T, H, W) tensor of spatial inputs (images or latent representations of images)
            sigma: (B,) tensor of noise standard deviations
            y_feat: List((B, L, y_feat_dim) tensor of caption token features. For SDXL text encoders: L=77, y_feat_dim=2048)
            y_mask: List((B, L) boolean tensor indicating which tokens are not padding)
            packed_indices: Dict with keys for Flash Attention. Result of compute_packed_indices.
        """
        
        T, ori_height, ori_width = x.shape[-3:]
        if ori_width % self.config.patch_size != 0:
            x = F.pad(x, (0, self.config.patch_size - ori_width % self.config.patch_size))
        if ori_height % self.config.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.config.patch_size - ori_height % self.config.patch_size))
        
        B, _, T, H, W = x.shape
        
        sigma = timestep / 1000
        # Use EFFICIENT_ATTENTION backend for T5 pooling, since we have a mask.
        # Have to call sdpa_kernel outside of a torch.compile region.
        
        # with sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
        x, c, y_feat, rope_cos, rope_sin = self.prepare(
            x, sigma, y_feat[0], y_mask[0]
        )
        # del y_mask

        N = x.size(1)

        attn_mask = y_mask[0][:, None, None, :].bool() # B, 1, 1, L
        all_one_mask = torch.ones(B, 1, 1, N, device=attn_mask.device, dtype=bool)
        attn_mask = torch.cat([all_one_mask, attn_mask], dim=-1)
        attn_mask = attn_mask.repeat(1, 1, attn_mask.shape[-1], 1)
        #B, 1, N+L, N+L
        
        assert attn_mask.dtype == torch.bool
        
        for i, block in enumerate(self.blocks):
            x, y_feat = auto_grad_checkpoint(block, x, c, y_feat, rope_cos, rope_sin, attn_mask)            
        # del y_feat  # Final layers don't use dense text features.

        x = self.final_layer(x, c)  # (B, M, patch_size ** 2 * out_channels)

        x = rearrange(
            x,
            "B (T hp wp) (p1 p2 c) -> B c T (hp p1) (wp p2)",
            T=T,
            hp=H // self.patch_size,
            wp=W // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels,
        )
        x = x[:, :, :, :ori_height, :ori_width]
        return x


@MODELS.register_module("Mochi1")
def mochi1_dit(from_pretrained=None, **kwargs):
    force_huggingface = kwargs.pop("force_huggingface", False)
    if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
        model = AsymmDiTJoint.from_pretrained(from_pretrained, **kwargs)
    else:
        config = AsymmDiTJointConfig(**kwargs)
        model = AsymmDiTJoint(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model