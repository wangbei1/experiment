import os
import torch
from vidgen.registry import MODELS
from .autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from vidgen.utils.misc import get_logger

@MODELS.register_module()
def Hunyuan_VAE(
    from_pretrained,
):
    config = AutoencoderKLCausal3D.load_config(from_pretrained)
    vae = AutoencoderKLCausal3D.from_config(config)
    if os.path.exists(os.path.join(from_pretrained, "pytorch_model.pt")):
        ckpt = torch.load(os.path.join(from_pretrained, "pytorch_model.pt"), map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if any(k.startswith("vae.") for k in ckpt.keys()):
            ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items() if k.startswith("vae.")}
        missing_keys, unexpected_keys = vae.load_state_dict(ckpt, strict=False)
        get_logger().info("Missing keys: %s", missing_keys)
        get_logger().info("Unexpected keys: %s", unexpected_keys)
        get_logger().info(f"Loaded vae checkpoint from {os.path.join(from_pretrained, 'pytorch_model.pt')}")
    else:
        get_logger()("Vae checkpoint is not loaded")
    return vae