import os
import torch
from .spatial_temporal_vae import SpatialTemporalVAE, SpatialTemporalVAEConfig
from vidgen.registry import MODELS
from vidgen.utils.ckpt_utils import load_checkpoint

@MODELS.register_module()
def JINGKAI_VAE(
    pretrained_spatial_vae_path,
    from_pretrained,
    force_huggingface=False,
    **kwargs
):
    
    if force_huggingface or (from_pretrained is not None and not os.path.exists(from_pretrained)):
        model = SpatialTemporalVAE.from_pretrained(from_pretrained, **kwargs)
    else:
        config = SpatialTemporalVAEConfig(pretrained_spatial_vae_path=pretrained_spatial_vae_path, **kwargs)
        model = SpatialTemporalVAE(config)

        if from_pretrained:
            load_checkpoint(model, from_pretrained)
    return model