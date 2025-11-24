from typing import Any, Dict, List, Optional, Union, Tuple
import os
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F

from .attention_processor import Attention, FluxAttnProcessor2_0, FluxSingleAttnProcessor2_0
from .activations import GEGLU, GELU, ApproximateGELU, SwiGLU
from .normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from .embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings

from vidgen.acceleration.checkpoint import auto_grad_checkpoint
from vidgen.registry import MODELS
from vidgen.utils.ckpt_utils import load_checkpoint
from vidgen.utils.train_utils import unmask_tokens

from transformers import PretrainedConfig, PreTrainedModel

def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


# YiYi to-do: refactor rope related functions/classes
class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)

class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states

class FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        processor = FluxSingleAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        enable_ffn_only_checkpoint=False,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        if enable_ffn_only_checkpoint:
            mlp_hidden_states = self.act_mlp(auto_grad_checkpoint(self.proj_mlp, norm_hidden_states))
        else:
            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        if enable_ffn_only_checkpoint:
            hidden_states = gate * auto_grad_checkpoint(self.proj_out, hidden_states)
        else:
            hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        enable_ffn_only_checkpoint=False,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if enable_ffn_only_checkpoint:
            ff_output = auto_grad_checkpoint(self.ff, norm_hidden_states)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # context_ff_output = self.ff_context(norm_encoder_hidden_states)
        context_ff_output = auto_grad_checkpoint(self.ff_context,norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states

class Linear_mapper(nn.Module):
    def __init__(self,in_dim=4096,out_dim=4096):
        super(Linear_mapper, self).__init__()
        self.fc1 = nn.Linear(in_dim, 2048)
        self.norm1 = nn.LayerNorm(2048)
        self.fc2 = nn.Linear(2048, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        self.norm3 = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        #x = torch.relu(x)  # 可以选择其他激活函数
        x = F.gelu(x.to(dtype=torch.float32), approximate="tanh").to(dtype=x.dtype)
        x = self.fc2(x)
        x = self.norm2(x)
        #x = torch.relu(x)  # 可以选择其他激活函数
        x = F.gelu(x.to(dtype=torch.float32), approximate="tanh").to(dtype=x.dtype)
        x = self.fc3(x)
        x = self.norm3(x)
        #x = torch.relu(x)  # 可以选择其他激活函数
        x = F.gelu(x.to(dtype=torch.float32), approximate="tanh").to(dtype=x.dtype)
        return x
    
class FluxTransformer3DModelConfig(PretrainedConfig):
    model_type = "FluxTransformer3DModel"

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: List[int] = [16, 56, 56],
        maskdit=False,
        decoder_depth=0,
        t5_embedder=False,
        **kwargs
    ):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.guidance_embeds = guidance_embeds
        self.axes_dims_rope=axes_dims_rope
        self.maskdit = maskdit
        self.decoder_depth = decoder_depth
        self.t5_embedder = t5_embedder
        super().__init__(**kwargs)

class FluxTransformer3DModel(PreTrainedModel):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    config_class = FluxTransformer3DModelConfig

    def __init__(self, config):
        super().__init__(config)
        
        self.out_channels = config.in_channels
        self.inner_dim = config.num_attention_heads * config.attention_head_dim

        self.pos_embed = EmbedND(dim=self.inner_dim, theta=10000, axes_dim=config.axes_dims_rope)
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if config.guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=config.pooled_projection_dim
        )

        self.context_embedder = nn.Linear(config.joint_attention_dim, self.inner_dim)
        
        if config.t5_embedder:
            self.t5_adapter = Linear_mapper(config.joint_attention_dim, config.joint_attention_dim)
        
        self.x_embedder = torch.nn.Linear(config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                )
                for i in range(config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                )
                for i in range(config.num_single_layers)
            ]
        )
        
        self.maskdit = config.maskdit
        if self.maskdit:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, config.attention_head_dim*config.num_attention_heads))
            nn.init.normal_(self.mask_token, std=.02)

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=True)

    def _pack_latents(self, latents):
        batch_size, num_channels_latents, temporal_size, height, width = latents.shape
        patch_size = self.config.patch_size
        
        latents = latents.view(batch_size, num_channels_latents, temporal_size, height // patch_size, patch_size, width // patch_size, patch_size)
        latents = latents.permute(0, 2, 3, 5, 1, 4, 6)
        latents = latents.reshape(batch_size, temporal_size * (height // patch_size) * (width // patch_size), num_channels_latents * patch_size * patch_size)

        return latents
    
    def _unpack_latents(self, latents, temporal_size, height, width):
        batch_size, num_patches, channels = latents.shape
        patch_size = self.config.patch_size
        latents = latents.view(batch_size, temporal_size, height // patch_size, width // patch_size, channels // (patch_size * patch_size), patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 2, 5, 3, 6)

        latents = latents.reshape(batch_size, channels // (patch_size * patch_size), temporal_size, height, width)

        return latents
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        guidance: torch.Tensor = None,
        xt_mask=None, xs_mask=None,
        **kwargs
    ):
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        
        T, ori_height, ori_width = hidden_states.shape[-3:]
        if ori_width % self.config.patch_size != 0:
            hidden_states = F.pad(hidden_states, (0, self.config.patch_size - ori_width % self.config.patch_size))
        if ori_height % self.config.patch_size != 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, self.config.patch_size - ori_height % self.config.patch_size))
        T, height, width = hidden_states.shape[-3:]
        
        hidden_states = self._pack_latents(hidden_states)
        
        # enable_sp
        enable_sp = "enable_sp" in kwargs
        if xs_mask is None and enable_sp:
            from training_acc.patches import prepare_flux_padding_inputs
            from training_acc.logger import logger#, log_rank
            _, seqlen, _ = hidden_states.shape
            # logger.info(log_rank(f"hidden_states shape:{hidden_states.shape}"))
            hidden_states = prepare_flux_padding_inputs(hidden_states)
            # logger.info(log_rank(f"local hidden_states shape:{hidden_states.shape}"))
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype)# * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        elif isinstance(self.time_text_embed, CombinedTimestepGuidanceTextProjEmbeddings):
            guidance = torch.ones_like(timestep) * 1000
        
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        
        if self.config.t5_embedder:
            encoder_hidden_states = self.t5_adapter(encoder_hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        #construct rope emb
        batch_size = encoder_hidden_states.shape[0]
        txt_ids = torch.zeros(batch_size, encoder_hidden_states.shape[1], 3, device=hidden_states.device, dtype=hidden_states.dtype)
        latent_image_ids = torch.zeros(T, height // self.config.patch_size, width // self.config.patch_size, 3, device=hidden_states.device, dtype=hidden_states.dtype)
        latent_image_ids[..., 0] = latent_image_ids[..., 0] + torch.arange(T, device=hidden_states.device, dtype=hidden_states.dtype)[:, None, None]
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // self.config.patch_size, device=hidden_states.device, dtype=hidden_states.dtype)[None, :, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // self.config.patch_size, device=hidden_states.device, dtype=hidden_states.dtype)[None, None, :]

        latent_image_id_t, latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_t * latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        latent_image_ids_orig = latent_image_ids.clone()
        if xs_mask is not None:
            B, T, S = xs_mask["ids_keep"].shape
            S_orig = hidden_states.shape[1]//T
            # mask hidden states
            hidden_states = rearrange(hidden_states, "B (T S) C -> B T S C", T=T, S=S_orig)
            hidden_states = torch.gather(hidden_states, dim=2, index=xs_mask["ids_keep"].unsqueeze(-1).repeat(1, 1, 1, hidden_states.shape[-1]))
            hidden_states = rearrange(hidden_states, "B T S C -> B (T S) C", T=T, S=S)
            # mask rope embedding
            latent_image_ids = latent_image_ids.reshape(batch_size, latent_image_id_t, latent_image_id_height * latent_image_id_width, latent_image_id_channels)
            latent_image_ids = torch.gather(latent_image_ids, dim=2, index=xs_mask["ids_keep"].unsqueeze(-1).repeat(1, 1, 1, latent_image_ids.shape[-1]))
            latent_image_ids = latent_image_ids.reshape(batch_size, -1, latent_image_id_channels)
        
        if xs_mask is not None and enable_sp:
            from training_acc.patches import prepare_flux_padding_inputs
            from training_acc.logger import logger, log_rank
            _, seqlen, _ = hidden_states.shape
            # logger.info(log_rank(f"hidden_states shape:{hidden_states.shape}"))
            hidden_states = prepare_flux_padding_inputs(hidden_states)
            # logger.info(log_rank(f"local hidden_states shape:{hidden_states.shape}"))
                
        if enable_sp:
            # logger.info(log_rank(f"latent_image_ids shape:{latent_image_ids.shape}"))  
            latent_image_ids = prepare_flux_padding_inputs(latent_image_ids) 
            # logger.info(log_rank(f"local latent_image_ids shape:{latent_image_ids.shape}"))  

        ids = torch.cat((txt_ids, latent_image_ids), dim=1)
        image_rotary_emb = self.pos_embed(ids)
        
        enable_ffn_only_checkpoint = kwargs.get("enable_ffn_only_checkpoint", False)
        for index_block, block in enumerate(self.transformer_blocks):
            if enable_ffn_only_checkpoint:
                # use ffn only ckpt, disable transformer-block-wise checkpointing, especially attention
                setattr(block, "grad_checkpointing", False)                
            encoder_hidden_states, hidden_states = auto_grad_checkpoint(block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, enable_ffn_only_checkpoint)

        enc_token_num = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        decoder_depth = self.config.decoder_depth if self.maskdit else 0
        for index_block, block in enumerate(self.single_transformer_blocks[:self.config.num_single_layers-decoder_depth]):
            if enable_ffn_only_checkpoint:
                setattr(block, "grad_checkpointing", False)
            hidden_states = auto_grad_checkpoint(block, hidden_states, temb, image_rotary_emb, enable_ffn_only_checkpoint)
        
        if xs_mask is not None:
            # unmask hidden states
            encoder_hidden_states, hidden_states = hidden_states[:,:enc_token_num], hidden_states[:,enc_token_num:]
            
            if enable_sp:
                from training_acc.patches import prepare_flux_padding_outputs
                # logger.info(log_rank(f"local output shape:{output.shape}"))  
                hidden_states = prepare_flux_padding_outputs(hidden_states, seqlen)
                # logger.info(log_rank(f"output shape:{output.shape}"))  
            
            hidden_states = unmask_tokens(hidden_states, xs_mask["ids_restore"], self.mask_token)
            S = hidden_states.shape[1]//T
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            # unmask rope embedding
            ids = torch.cat((txt_ids, latent_image_ids_orig), dim=1)
            image_rotary_emb = self.pos_embed(ids)
            
        for index_block, block in enumerate(self.single_transformer_blocks[self.config.num_single_layers-decoder_depth:]):
            if enable_ffn_only_checkpoint:
                setattr(block, "grad_checkpointing", False)                
            hidden_states = auto_grad_checkpoint(block, hidden_states, temb, image_rotary_emb, enable_ffn_only_checkpoint)

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        
        if xs_mask is None and enable_sp:
            from training_acc.patches import prepare_flux_padding_outputs
            # logger.info(log_rank(f"local output shape:{output.shape}"))  
            output = prepare_flux_padding_outputs(output, seqlen)
            # logger.info(log_rank(f"output shape:{output.shape}"))  
        output = self._unpack_latents(output, T, height, width)
        output = output[:, :, :, :ori_height, :ori_width]

        return output
    
@MODELS.register_module("Flux-3D")
def Flux_3d(from_pretrained=None, **kwargs):
    force_huggingface = kwargs.pop("force_huggingface", False)
    if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
        model = FluxTransformer3DModel.from_pretrained(from_pretrained, **kwargs)
    else:
        config = FluxTransformer3DModelConfig(**kwargs)
        model = FluxTransformer3DModel(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model