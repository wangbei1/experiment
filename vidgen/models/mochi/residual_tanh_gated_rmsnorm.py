import torch


class ResidualTanhGatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_res, gate, eps=1e-6):
        # Convert to fp32 for precision
        x_res_fp32 = x_res.float()

        # Compute RMS
        mean_square = x_res_fp32.pow(2).mean(-1, keepdim=True)
        scale = torch.rsqrt(mean_square + eps)

        # Apply tanh to gate
        tanh_gate = torch.tanh(gate).unsqueeze(1)

        # Normalize and apply gated scaling
        x_normed = x_res_fp32 * scale * tanh_gate

        ctx.save_for_backward(x, x_res, gate, scale, tanh_gate)
        ctx.eps = eps
        
        # Apply residual connection
        output = x + x_normed.type_as(x)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, x_res, gate, scale, tanh_gate = ctx.saved_tensors
        eps = ctx.eps
        
        # Compute gradients for x and x_res
        grad_x = grad_output.clone()
        grad_x_res = grad_output * scale * tanh_gate.type_as(grad_output)
        
        # Compute gradients for gate
        x_res_fp32 = x_res.float()
        d_tanh_gate = (1 - tanh_gate.pow(2)).squeeze(1)
        grad_gate = (grad_output * (x_res_fp32 * scale)).sum(dim=-1) * d_tanh_gate
        
        return grad_x, grad_x_res, grad_gate, None


def residual_tanh_gated_rmsnorm(x, x_res, gate, eps=1e-6):
    # return ResidualTanhGatedRMSNorm.apply(x, x_res, gate, eps)
    x_res_fp32 = x_res.float()

    # Compute RMS
    mean_square = x_res_fp32.pow(2).mean(-1, keepdim=True)
    scale = torch.rsqrt(mean_square + eps)

    # Apply tanh to gate
    tanh_gate = torch.tanh(gate).unsqueeze(1)

    # Normalize and apply gated scaling
    x_normed = x_res_fp32 * scale * tanh_gate

    # Apply residual connection
    output = x + x_normed.type_as(x)

    return output
