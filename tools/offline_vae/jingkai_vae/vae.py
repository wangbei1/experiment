import os
import torch
from .spatial_temporal_vae import SpatialTemporalVAE, SpatialTemporalVAEConfig

def JINGKAI_VAE(
    pretrained_spatial_vae_path,
    from_pretrained,
    **kwargs
):
    
    config = SpatialTemporalVAEConfig(pretrained_spatial_vae_path=pretrained_spatial_vae_path, **kwargs)
    model = SpatialTemporalVAE(config)

    if from_pretrained:
        print("Load from pretrained: ", from_pretrained)
        state_dict = torch.load(from_pretrained)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("Missing keys: %s", missing_keys)
        print("Unexpected keys: %s", unexpected_keys)
    return model