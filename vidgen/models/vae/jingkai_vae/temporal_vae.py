import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def pad_at_dim(t, pad, dim=-1, pad_mode="replicate"):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), mode=pad_mode)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        strides=None,  # allow custom stride
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = strides[0] if strides is not None else kwargs.pop("stride", 1)

        time_padding = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_padding = time_padding
        self.spatial_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)

        stride = strides if strides is not None else (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

        self.conv_cache = None

    def pad_at_time(self, x):
        if self.time_padding > 0:
            if self.conv_cache is not None:
                cached_inputs = [self.conv_cache.to(dtype=x.dtype, device=x.device)]
                x = torch.cat(cached_inputs + [x], dim=2)
            else:
                x = F.pad(x, (0, 0, 0, 0, self.time_padding, 0), mode="replicate")
        return x

    def clear_causal_conv_cache(self):
        del self.conv_cache
        self.conv_cache = None

    def forward(self, x):
        x = self.pad_at_time(x)

        self.clear_causal_conv_cache()
        # Note: we could move these to the cpu for a lower maximum memory usage but its only a few
        # hundred megabytes and so let's not do it for now
        self.conv_cache = x[:, :, -self.time_padding:].clone()

        x = F.pad(x, self.spatial_padding, mode="constant", value=0.)
        x = self.conv(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        dropout=0.,
        max_len=32,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CausalAttn(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        max_len=32,
    ):
        super().__init__()
        self.trans = BasicTransformerBlock(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
        )
        self.pe = PositionalEncoding(dim, max_len=max_len)

    @staticmethod
    def get_causal_mask(hidden_states):
        batch_size, sequence_length, dim = hidden_states.shape
        mask = torch.tril(torch.ones(sequence_length, sequence_length))
        # generate attention mask from binary values
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0)
        mask = mask.repeat(batch_size, 1, 1)
        return mask.to(device=hidden_states.device, dtype=hidden_states.dtype)

    def forward(self, hidden_states):
        h, w = hidden_states.shape[-2:]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b h w) f c")
        hidden_states = self.pe(hidden_states)
        causal_mask = self.get_causal_mask(hidden_states)
        hidden_states = self.trans(
            hidden_states=hidden_states,
            attention_mask=causal_mask,
        )
        hidden_states = rearrange(
            hidden_states,
            "(b h w) f c -> b c f h w",
            h=h, w=w,
        )
        return hidden_states


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,  # SCH: added
        filters,
        conv_fn,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
        num_groups=32,
        add_causal_attn=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.activate = activation_fn()
        self.use_conv_shortcut = use_conv_shortcut

        # SCH: MAGVIT uses GroupNorm by default
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = conv_fn(in_channels, self.filters, kernel_size=(3, 3, 3), bias=False)
        self.norm2 = nn.GroupNorm(num_groups, self.filters)
        self.conv2 = conv_fn(self.filters, self.filters, kernel_size=(3, 3, 3), bias=False)
        if in_channels != filters:
            if self.use_conv_shortcut:
                self.conv3 = conv_fn(in_channels, self.filters, kernel_size=(3, 3, 3), bias=False)
            else:
                self.conv3 = conv_fn(in_channels, self.filters, kernel_size=(1, 1, 1), bias=False)

        self.causal_attn = CausalAttn(
            dim=self.filters,
            num_attention_heads=8,
            attention_head_dim=self.filters//8,
        ) if add_causal_attn else None

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        if self.in_channels != self.filters:  # SCH: ResBlock X->Y
            residual = self.conv3(residual)
        x = x + residual
        if self.causal_attn is not None:
            x = self.causal_attn(x)
        return x


def get_activation_fn(activation):
    activation = activation.lower()
    if activation == "relu":
        activation_fn = nn.ReLU
    elif activation in ("swish", "silu"):
        activation_fn = nn.SiLU
    else:
        raise NotImplementedError
    return activation_fn


class Encoder(nn.Module):
    """Encoder Blocks."""

    def __init__(
        self,
        in_out_channels=16,
        latent_embed_dim=16,  # num channels for latent vector
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
        mid_block_add_attention=False,
    ):
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim

        self.activation_fn = get_activation_fn(activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = CausalConv3d
        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )

        # first layer conv
        self.conv_in = self.conv_fn(
            in_out_channels,
            filters,
            kernel_size=(3, 3, 3),
        )

        # ResBlocks and conv downsample
        self.block_res_blocks = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        filters = self.filters
        prev_filters = filters  # record for in_channels
        for i in range(self.num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            block_items = nn.ModuleList([])
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # update in_channels
            self.block_res_blocks.append(block_items)

            if i < self.num_blocks - 1:
                if self.temporal_downsample[i]:
                    t_stride = 2 if self.temporal_downsample[i] else 1
                    s_stride = 1
                    self.conv_blocks.append(
                        self.conv_fn(
                            prev_filters,
                            filters,
                            kernel_size=(3, 3, 3),
                            strides=(t_stride, s_stride, s_stride),
                        )
                    )
                    prev_filters = filters  # update in_channels
                else:
                    # if no t downsample, don't add since this does nothing for pipeline models
                    self.conv_blocks.append(nn.Identity(prev_filters))  # Identity
                    prev_filters = filters  # update in_channels

        # mid blocks
        self.mid_blocks = nn.ModuleList([])
        for _ in range(self.num_res_blocks):
            self.mid_blocks.append(ResBlock(
                filters, filters, add_causal_attn=mid_block_add_attention, **self.block_args
            ))

        # MAGVIT uses Group Normalization
        self.norm1 = nn.GroupNorm(self.num_groups, prev_filters)

        self.conv2 = self.conv_fn(
            prev_filters,
            self.embedding_dim,
            kernel_size=(1, 1, 1),
        )

    def forward(self, x):
        x = self.conv_in(x)

        for i in range(self.num_blocks):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i < self.num_blocks - 1:
                x = self.conv_blocks[i](x)
        for i in range(self.num_res_blocks):
            x = self.mid_blocks[i](x)

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    """Decoder Blocks."""

    def __init__(
        self,
        in_out_channels=16,
        latent_embed_dim=16,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
        mid_block_add_attention=False,
    ):
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim
        self.s_stride = 1

        self.activation_fn = get_activation_fn(activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = CausalConv3d
        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )

        filters = self.filters * self.channel_multipliers[-1]
        prev_filters = filters

        # last conv
        self.conv1 = self.conv_fn(
            self.embedding_dim,
            filters,
            kernel_size=(3, 3, 3),
        )

        # last layer res block
        self.mid_blocks = nn.ModuleList([])
        for _ in range(self.num_res_blocks):
            self.mid_blocks.append(ResBlock(
                filters, filters, add_causal_attn=mid_block_add_attention, **self.block_args
            ))

        # ResBlocks and conv upsample
        self.block_res_blocks = nn.ModuleList([])
        self.num_blocks = len(self.channel_multipliers)
        self.conv_blocks = nn.ModuleList([])
        # reverse to keep track of the in_channels, but append also in a reverse direction
        for i in reversed(range(self.num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            # resblock handling
            block_items = nn.ModuleList([])
            for _ in range(self.num_res_blocks):
                block_items.append(ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # SCH: update in_channels
            self.block_res_blocks.insert(0, block_items)  # SCH: append in front

            # conv blocks with upsampling
            if i > 0:
                if self.temporal_downsample[i - 1]:
                    t_stride = 2 if self.temporal_downsample[i - 1] else 1
                    # SCH: T-Causal Conv 3x3x3, f -> (t_stride * 2 * 2) * f, depth to space t_stride x 2 x 2
                    self.conv_blocks.insert(
                        0,
                        self.conv_fn(
                            prev_filters,
                            prev_filters * t_stride * self.s_stride * self.s_stride,
                            kernel_size=(3, 3, 3),
                        ),
                    )
                else:
                    self.conv_blocks.insert(
                        0,
                        nn.Identity(prev_filters),
                    )

        self.norm1 = nn.GroupNorm(self.num_groups, prev_filters)

        self.conv_out = self.conv_fn(filters, in_out_channels, 3)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(self.num_res_blocks):
            x = self.mid_blocks[i](x)
        for i in reversed(range(self.num_blocks)):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i > 0:
                t_stride = 2 if self.temporal_downsample[i - 1] else 1
                x = self.conv_blocks[i - 1](x)
                x = rearrange(
                    x,
                    "B (C ts hs ws) T H W -> B C (T ts) (H hs) (W ws)",
                    ts=t_stride,
                    hs=self.s_stride,
                    ws=self.s_stride,
                )

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv_out(x)
        return x


class TemporalVAE(nn.Module):
    def __init__(
        self,
        in_out_channels=512,
        latent_embed_dim=16,
        embed_dim=16,
        filters=512,
        num_res_blocks=4,
        channel_multipliers=(1, 1, 1, 1),
        temporal_downsample=(False, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
        scaling_factor=1,
        shift_factor=0,
        latents_mean=None,
        latents_std=None,
        use_quant_conv=True,
        use_post_quant_conv=True,
        mid_block_add_attention=False,
    ):
        super().__init__()

        self.time_downsample_factor = 2 ** sum(temporal_downsample)
        self.patch_size = (self.time_downsample_factor, 1, 1)
        self.out_channels = in_out_channels

        # NOTE: following MAGVIT, conv in bias=False in encoder first conv
        self.encoder = Encoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim * 2,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.quant_conv = CausalConv3d(2 * latent_embed_dim, 2 * embed_dim, 1)

        self.post_quant_conv = CausalConv3d(embed_dim, latent_embed_dim, 1)
        self.decoder = Decoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
            mid_block_add_attention=mid_block_add_attention,
        )
        self.shift_factor = shift_factor
        self.scaling_factor = scaling_factor
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        self.use_quant_conv = use_quant_conv
        self.use_post_quant_conv = use_post_quant_conv
        self.time_padding = self.calculate_time_padding()

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            if input_size[i] is None:
                lsize = None
            elif i == 0:
                time_padding = (
                    0
                    if (input_size[i] % self.time_downsample_factor == 0)
                    else self.time_downsample_factor - input_size[i] % self.time_downsample_factor
                )
                lsize = (input_size[i] + time_padding) // self.patch_size[i]
            else:
                lsize = input_size[i] // self.patch_size[i]
            latent_size.append(lsize)
        return latent_size

    def _encode(self, x, time_slice=-1):
        if 0 < time_slice < self.time_padding * self.time_downsample_factor:
            print(f"WARNING: `time_slice` ({time_slice}) < "
                  f"self.time_padding ({self.time_padding}) * "
                  f"self.time_downsample_factor ({self.time_downsample_factor}), "
                  f"which will not be used!!!")
        if time_slice >= self.time_padding * self.time_downsample_factor:
            time_slice = max(
                self.time_padding * self.time_downsample_factor,
                round(time_slice / self.time_downsample_factor) * self.time_downsample_factor,
            )
            moments_slices = []
            for i in range(0, x.shape[2], time_slice):
                x_slice = x[:, :, i:i + time_slice]
                moments_slices.append(
                    self.quant_conv(self.encoder(x_slice))
                )
            moments = torch.cat(moments_slices, dim=2)
        else:
            encoded_feature = self.encoder(x)
            moments = self.quant_conv(encoded_feature)

        self.clear_causal_conv_cache()
        return moments

    def encode(self, x, batch_slice=-1, time_slice=-1):
        time_padding = (
            0
            if (x.shape[2] % self.time_downsample_factor == 0)
            else self.time_downsample_factor - x.shape[2] % self.time_downsample_factor
        )

        x = pad_at_dim(x, (time_padding, 0), dim=2)
        if batch_slice > 0:
            moments_slices = []
            for i in range(0, x.shape[0], batch_slice):
                x_slice = x[i:i + batch_slice]
                moments_slices.append(
                    self._encode(x_slice, time_slice)
                )
            moments = torch.cat(moments_slices, dim=0)
        else:
            moments = self._encode(x, time_slice)
        posterior = DiagonalGaussianDistribution(moments.to(x.dtype))
        return posterior

    def _decode(self, z, time_slice=-1):
        time_slice = round(time_slice)
        if 0 < time_slice < self.time_padding:
            print(f"WARNING: `time_slice` ({time_slice}) < "
                  f"self.time_padding ({self.time_padding}), "
                  f"which will not be used!!!")
        if time_slice >= self.time_padding:
            x_slices = []
            for i in range(0, z.shape[2], time_slice):
                z_slice = z[:, :, i:i + time_slice]
                x_slices.append(
                    self.decoder(self.post_quant_conv(z_slice))
                )
            x = torch.cat(x_slices, dim=2)
        else:
            z = self.post_quant_conv(z)
            x = self.decoder(z)

        self.clear_causal_conv_cache()
        return x

    def decode(self, z, num_frames=None, batch_slice=-1, time_slice=-1):
        time_padding = (
            0
            if (num_frames % self.time_downsample_factor == 0)
            else self.time_downsample_factor - num_frames % self.time_downsample_factor
        )

        if batch_slice > 0:
            x_slices = []
            for i in range(0, z.shape[0], batch_slice):
                z_slice = z[i:i + batch_slice]
                x_slices.append(
                    self._decode(z_slice, time_slice)
                )
            x = torch.cat(x_slices, dim=0)
        else:
            x = self._decode(z, time_slice)

        x = x[:, :, time_padding:]
        return x

    def forward(self, x, sample_posterior=True, encode_time_slice=-1, decode_time_slice=-1):
        num_frames = x.shape[2]
        posterior = self.encode(x, time_slice=encode_time_slice)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        recon_video = self.decode(z, num_frames=num_frames, time_slice=decode_time_slice)
        return recon_video, posterior, z

    def clear_causal_conv_cache(self):
        for name, module in self.named_modules():
            if isinstance(module, CausalConv3d):
                module.clear_causal_conv_cache()

    def calculate_time_padding(self):
        time_padding = 0
        for name, module in self.named_modules():
            if isinstance(module, CausalConv3d) and module.time_padding > time_padding:
                time_padding = module.time_padding
        return time_padding
