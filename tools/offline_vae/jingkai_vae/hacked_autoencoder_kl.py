from typing import Optional, Union, Any

import torch

from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils import is_torch_version


def _encode_emb(model, sample: torch.Tensor) -> torch.Tensor:
    r"""Copied from the forward method of the `Encoder` class in diffusers==0.30.0.dev0."""

    sample = model.conv_in(sample)

    if model.training and model.gradient_checkpointing:

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        # down
        if is_torch_version(">=", "1.11.0"):
            for down_block in model.down_blocks:
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block), sample, use_reentrant=False
                )
            # middle
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(model.mid_block), sample, use_reentrant=False
            )
        else:
            for down_block in model.down_blocks:
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
            # middle
            sample = torch.utils.checkpoint.checkpoint(create_custom_forward(model.mid_block), sample)

    else:
        # down
        for down_block in model.down_blocks:
            sample = down_block(sample)

        # middle
        sample = model.mid_block(sample)

    # post-process
    sample = model.conv_norm_out(sample)
    sample = model.conv_act(sample)
    # ##################################################
    # Skip the conv_out in encode_emb
    # ##################################################
    # sample = model.conv_out(sample)

    return sample


def _decode_emb(model, sample: torch.Tensor, latent_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""The forward method of the `Decoder` class in diffusers==0.30.0.dev0."""

    # ##################################################
    # Skip the conv_in in decode_emb
    # ##################################################
    # sample = model.conv_in(sample)

    upscale_dtype = next(iter(model.up_blocks.parameters())).dtype
    if model.training and model.gradient_checkpointing:

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        if is_torch_version(">=", "1.11.0"):
            # middle
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(model.mid_block),
                sample,
                latent_embeds,
                use_reentrant=False,
            )
            sample = sample.to(upscale_dtype)

            # up
            for up_block in model.up_blocks:
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(up_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
        else:
            # middle
            sample = torch.utils.checkpoint.checkpoint(
                create_custom_forward(model.mid_block), sample, latent_embeds
            )
            sample = sample.to(upscale_dtype)

            # up
            for up_block in model.up_blocks:
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
    else:
        # middle
        sample = model.mid_block(sample, latent_embeds)
        sample = sample.to(upscale_dtype)

        # up
        for up_block in model.up_blocks:
            sample = up_block(sample, latent_embeds)

    # post-process
    if latent_embeds is None:
        sample = model.conv_norm_out(sample)
    else:
        sample = model.conv_norm_out(sample, latent_embeds)
    sample = model.conv_act(sample)
    sample = model.conv_out(sample)

    return sample


class HackedAutoencoderKL(AutoencoderKL):
    def get_training_emb(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Union[torch.Tensor, Any], Union[torch.Tensor, Any]]:
        sample_emb = self.encode_emb(sample)

        moment = self.encoder.conv_norm_out(sample_emb)
        moment = self.encoder.conv_act(moment)
        moment = self.encoder.conv_out(moment)
        if self.quant_conv is not None:
            moment = self.quant_conv(moment)

        posterior = DiagonalGaussianDistribution(moment)

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        target_emb = self.decoder.conv_in(z)
        return sample_emb, target_emb

    def encode_emb(self, sample: torch.Tensor) -> torch.Tensor:
        return _encode_emb(self.encoder, sample)

    def decode_emb(self, sample: torch.Tensor, latent_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        return _decode_emb(self.decoder, sample, latent_embeds)

