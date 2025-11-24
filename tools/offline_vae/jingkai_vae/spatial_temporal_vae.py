from dataclasses import dataclass
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn

from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils import BaseOutput

from .hacked_autoencoder_kl import HackedAutoencoderKL
from .temporal_vae import TemporalVAE
from transformers import PretrainedConfig, PreTrainedModel
from einops import rearrange

def proc_3d_using_2d_func(x, func, slice_size=-1):
    num_frames = x.shape[2]
    x = rearrange(x, "b c f h w -> (b f) c h w")
    if slice_size > 0:
        slices = []
        for i in range(0, x.shape[0], slice_size):
            x_slice = x[i:i+slice_size]
            slices.append(func(x_slice))
        x = torch.cat(slices, dim=0)
    else:
        x = func(x)
    return rearrange(x, "(b f) c h w -> b c f h w", f=num_frames)

@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.Tensor
    commit_loss: Optional[torch.FloatTensor] = None
    posterior: Optional[torch.FloatTensor] = None


class SpatialTemporalVAEConfig(PretrainedConfig):
    model_type = "SpatialTemporalVAE"

    def __init__(
        self,
        pretrained_spatial_vae_path,
        in_out_channels=512,
        latent_embed_dim=16,
        embed_dim=16,
        filters=512,
        num_res_blocks=4,
        channel_multipliers = [1, 1, 1, 1],
        temporal_downsample= [False, True, True],
        num_groups=32,
        activation_fn="swish",
        mid_block_add_attention=False,
        scaling_factor=0.89381361,
        shift_factor=0.00566754,
        subfolder="vae",
        mode="emb_compression",
        **kwargs,
    ):
        self.pretrained_spatial_vae_path = pretrained_spatial_vae_path
        self.subfolder = subfolder
        self.mode = mode
        self.scaling_factor=scaling_factor
        self.shift_factor = shift_factor
        self.temporal_vae_kwargs = dict(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim,
            embed_dim=embed_dim,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            num_groups=num_groups,
            activation_fn=activation_fn,
            mid_block_add_attention=mid_block_add_attention
        )
        super().__init__(**kwargs)

        
class SpatialTemporalVAE(PreTrainedModel):
    config_class = SpatialTemporalVAEConfig

    def __init__(self, config: SpatialTemporalVAEConfig):
        super().__init__(config=config)
        pretrained_spatial_vae_path = config.pretrained_spatial_vae_path
        subfolder = config.subfolder
        temporal_vae_kwargs = config.temporal_vae_kwargs
        mode = config.mode
        
        self.latent_embed_dim = temporal_vae_kwargs['latent_embed_dim']
        
        mode = mode.lower()
        assert mode in ("emb_compression", "latent_compression", "no_compression")
        self.mode = mode

        if temporal_vae_kwargs is None:
            temporal_vae_kwargs = {}
        else:
            temporal_vae_kwargs = temporal_vae_kwargs.copy()

        self.spatial_vae = HackedAutoencoderKL.from_pretrained(
            pretrained_spatial_vae_path, subfolder=subfolder,
        )

        if self.mode == "no_compression":
            self.temporal_vae = None
        else:
            self.temporal_vae = TemporalVAE(
                **temporal_vae_kwargs
            )

        self.use_slicing = False

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    @property
    def scaling_factor(self):
        return self.temporal_vae.scaling_factorv

    @property
    def shift_factor(self):
        return self.temporal_vae.shift_factor

    @property
    def latents_mean(self):
        return self.temporal_vae.latents_mean
    @property
    def latents_std(self):
        return self.temporal_vae.latents_std

    @property
    def use_quant_conv(self):
        return self.temporal_vae.use_quant_conv

    @property
    def use_post_quant_conv(self):
        return self.temporal_vae.use_post_quant_con

    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        batch_slice_2d: int = -1,
        batch_slice_1d: int = -1,
        time_slice: int = -1,
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        # override use_slicing here
        if self.spatial_vae.use_slicing:
            self.use_slicing = True
            self.spatial_vae.use_slicing = False
        if self.use_slicing and batch_slice_2d <= 0:
            batch_slice_2d = 1

        if self.mode == "no_compression":
            h = proc_3d_using_2d_func(
                x,
                lambda sample: self.spatial_vae.encoder(sample),
                slice_size=batch_slice_2d,
            )
            if self.spatial_vae.quant_conv is not None:
                moments = self.spatial_vae.quant_conv(h)
            else:
                moments = h
            posterior = DiagonalGaussianDistribution(moments)
        elif self.mode == "latent_compression":
            latent = proc_3d_using_2d_func(
                x,
                lambda sample: self.spatial_vae.encode(sample).latent_dist.mode(),
                slice_size=batch_slice_2d,
            ).sub(self.spatial_vae.config.shift_factor).mul(self.spatial_vae.config.scaling_factor)
            posterior = self.temporal_vae.encode(latent, batch_slice=batch_slice_1d, time_slice=time_slice)
        elif self.mode == "emb_compression":
            emb = proc_3d_using_2d_func(
                x,
                lambda sample: self.spatial_vae.encode_emb(sample),
                slice_size=batch_slice_2d,
            )
            posterior = self.temporal_vae.encode(emb, batch_slice=batch_slice_1d, time_slice=time_slice)
        else:
            raise RuntimeError(f"Invalid mode {self.mode}")

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(
        self,
        z: torch.Tensor,
        num_frames: int,
        return_dict: bool = True,
        batch_slice_2d: int = -1,
        batch_slice_1d: int = -1,
        time_slice: int = -1,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        # override use_slicing here
        if self.spatial_vae.use_slicing:
            self.use_slicing = True
            self.spatial_vae.use_slicing = False
        if self.use_slicing and batch_slice_2d <= 0:
            batch_slice_2d = 1

        if self.mode == "no_compression":
            decoded = proc_3d_using_2d_func(
                z,
                lambda sample: self.spatial_vae.decode(sample, return_dict=False)[0],
                slice_size=batch_slice_2d,
            )
        elif self.mode == "latent_compression":
            latent = self.temporal_vae.decode(
                z, num_frames=num_frames, batch_slice=batch_slice_1d, time_slice=time_slice
            ).div(self.spatial_vae.config.scaling_factor).add(self.spatial_vae.config.shift_factor)
            decoded = proc_3d_using_2d_func(
                latent,
                lambda sample: self.spatial_vae.decode(sample, return_dict=False)[0],
                slice_size=batch_slice_2d,
            )
        elif self.mode == "emb_compression":
            emb = self.temporal_vae.decode(
                z, num_frames=num_frames, batch_slice=batch_slice_1d, time_slice=time_slice
            )
            decoded = proc_3d_using_2d_func(
                emb,
                lambda sample: self.spatial_vae.decode_emb(sample),
                slice_size=batch_slice_2d,
            )
        else:
            raise RuntimeError(f"Invalid mode {self.mode}")

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = True,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        batch_slice_2d: int = -1,
        batch_slice_1d: int = -1,
        enc_time_slice: int = -1,
        dec_time_slice: int = -1,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor, DiagonalGaussianDistribution]]:
        x = sample
        num_frames = x.shape[2]
        posterior = self.encode(
            x,
            batch_slice_2d=batch_slice_2d,
            batch_slice_1d=batch_slice_1d,
            time_slice=enc_time_slice
        ).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(
            z,
            num_frames,
            batch_slice_2d=batch_slice_2d,
            batch_slice_1d=batch_slice_1d,
            time_slice=dec_time_slice
        ).sample

        if not return_dict:
            return (dec, posterior,)

        return DecoderOutput(sample=dec, posterior=posterior)

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            if input_size[i] is None:
                lsize = None
            elif i == 0:
                time_padding = (
                    0
                    if (input_size[i] % self.temporal_vae.time_downsample_factor == 0)
                    else self.temporal_vae.time_downsample_factor - input_size[i] % self.temporal_vae.time_downsample_factor
                )
                lsize = (input_size[i] + time_padding) // self.temporal_vae.patch_size[i]
            else:
                vae_scale_factor_spatial = 2 ** (len(self.spatial_vae.config.block_out_channels) - 1)
                lsize = input_size[i] // vae_scale_factor_spatial
            latent_size.append(lsize)
        return latent_size