import torch


# class ModulatedRMSNorm(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, scale, eps=1e-6):
#         # Convert to fp32 for precision
#         x_fp32 = x.float()
#         scale_fp32 = scale.float()

#         # Compute RMS
#         mean_square = x_fp32.pow(2).mean(-1, keepdim=True)
#         inv_rms = torch.rsqrt(mean_square + eps)

#         # Normalize and modulate
#         x_normed = x_fp32 * inv_rms
#         x_modulated = x_normed * (1 + scale_fp32.unsqueeze(1))

#         return x_modulated.type_as(x)

class ModulatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, eps=1e-6):
        # Convert to fp32 for precision
        x_fp32 = x.float()
        scale_fp32 = scale.float()
        # Compute RMS
        mean_square = x_fp32.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + eps)
        # Normalize and modulate
        x_normed = x_fp32 * inv_rms
        x_modulated = x_normed * (1 + scale_fp32.unsqueeze(1))
        
        # Save tensors for backward pass
        ctx.save_for_backward(x_fp32, scale_fp32, inv_rms, x_normed)
        ctx.eps = eps

        return x_modulated.type_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        x_fp32, scale_fp32, inv_rms, x_normed = ctx.saved_tensors
        eps = ctx.eps

        # Calculate gradients
        grad_x_normed = grad_output * (1 + scale_fp32.unsqueeze(1))
        grad_inv_rms = (grad_x_normed * x_fp32).sum(-1, keepdim=True) * -0.5 * (inv_rms ** 3)
        
        # Gradient with respect to x
        grad_x = grad_x_normed * inv_rms + grad_inv_rms * 2 * x_fp32 / x_fp32.size(-1)
        
        # Gradient with respect to scale
        grad_scale = (grad_output * x_normed).sum(dim=-1)

        return grad_x.type_as(grad_output), grad_scale.type_as(grad_output), None

def modulated_rmsnorm(x, scale, eps=1e-6):
    # return ModulatedRMSNorm.apply(x, scale, eps)
    x_fp32 = x.float()
    scale_fp32 = scale.float()

    # Compute RMS
    mean_square = x_fp32.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + eps)

    # Normalize and modulate
    x_normed = x_fp32 * inv_rms
    x_modulated = x_normed * (1 + scale_fp32.unsqueeze(1))

    return x_modulated.type_as(x)
