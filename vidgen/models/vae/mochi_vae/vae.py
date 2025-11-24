import os
import torch
import torch.nn as nn
from vidgen.registry import MODELS
from vidgen.models.vae.mochi_vae.models import (
    Encoder,
    Decoder,
    decode_latents_tiled_full, 
    decode_latents_tiled_spatial,
    encode_latents_tiled_full, 
    encode_latents_tiled_temporal, 
    add_fourier_features
)
from vidgen.models.vae.mochi_vae.vae_stats import vae_latents_to_dit_latents, dit_latents_to_vae_latents
from safetensors.torch import load_file
from vidgen.models.vae.mochi_vae.utils import save_video

@MODELS.register_module()
class VideoAutoencoderPipeline(nn.Module):

    def __init__(self):
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Create VAE encoder
        self.encoder = Encoder(
            in_channels=15,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 6],
            num_res_blocks=[3, 3, 4, 6, 3],
            latent_dim=12,
            temporal_reductions=[1, 2, 3],
            spatial_reductions=[2, 2, 2],
            prune_bottlenecks=[False, False, False, False, False],
            has_attentions=[False, True, True, True, True],
            affine=True,
            bias=True,
            input_is_conv_1x1=True,
            padding_mode="replicate"
        )
        self.encoder = self.encoder.to(memory_format=torch.channels_last_3d)
        self.encoder.eval()
        
        self.downsample_factor = [6, 8, 8]
        self.latent_embed_dim = 12
        
        # Create VAE decoder
        self.decoder = Decoder(
            out_channels=3,
            base_channels=128,
            channel_multipliers=[1, 2, 4, 6],
            temporal_expansions=[1, 2, 3],
            spatial_expansions=[2, 2, 2],
            num_res_blocks=[3, 3, 4, 6, 3],
            latent_dim=12,
            has_attention=[False, False, False, False, False],
            output_norm=False,
            nonlinearity="silu",
            output_nonlinearity="silu",
            causal=True,
        )
        # VAE is not FSDP-wrapped
        self.decoder.eval()


    def encode(self, x):
        # from einops import rearrange
        # import numpy as np
        # tmp_x = (x.float().cpu().numpy()[0]+1.)*127.5
        # save_video(rearrange(tmp_x.astype(np.uint8), 'c t h w -> t h w c'), f"x_resize.mp4", fps=24)
        x = add_fourier_features(x)
        frame_batch_size = 12
        if x.shape[2] < frame_batch_size:
            x = self.encoder(x)
        else:
            x = encode_latents_tiled_temporal(self.encoder, x, frame_batch_size)
        x = self.encoder.latent_dist(x).sample()
        x = vae_latents_to_dit_latents(x)
        return x

    def decode(self, z, num_frames=None):
        z = dit_latents_to_vae_latents(z)
        # x_rec = decode_latents_tiled_full(self.decoder, z)
        x_rec = decode_latents_tiled_spatial(self.decoder, z, num_tiles_w=2, num_tiles_h=2, overlap=10)
        # x_rec = decode_latents(self.decoder, z)
        # save_video(x_rec.cpu().numpy()[0], f"x_recon.mp4", fps=24)
        return x_rec
        
    def load_pretrained_encoder(self, from_pretrained_encoder):
        self.encoder.load_state_dict(load_file(from_pretrained_encoder), strict=True)
        
    def load_pretrained_decoder(self, from_pretrained_decoder):
        self.decoder.load_state_dict(load_file(from_pretrained_decoder), strict=True)
        
    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            if input_size[i] is None:
                lsize = None
            elif i == 0:
                time_padding = (
                    0
                    if (input_size[i] % self.downsample_factor[0] == 0)
                    else self.downsample_factor[i] - input_size[i] % self.downsample_factor[i]
                )
                lsize = (input_size[i] + time_padding) // self.downsample_factor[i]
            else:
                lsize = input_size[i] // self.downsample_factor[i]
            latent_size.append(lsize)
        return latent_size
    
    
@MODELS.register_module("MochiVAE")
def MochiVAE(
    from_pretrained_encoder=None,
    from_pretrained_decoder=None,
    **kwargs
):
    model = VideoAutoencoderPipeline()

    if from_pretrained_encoder:
        model.load_pretrained_encoder(from_pretrained_encoder)
    
    if from_pretrained_decoder:
        model.load_pretrained_decoder(from_pretrained_decoder)
        
    return model
