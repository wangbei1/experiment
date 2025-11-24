import torch
from tqdm import tqdm

from vidgen.registry import SCHEDULERS

from .rectified_flow import RFlowScheduler, timestep_transform


@SCHEDULERS.register_module("rflow-mochi")
class RFLOW_MOCHI:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        use_fixed_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform
        self.use_fixed_timestep_transform = use_fixed_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            use_fixed_timestep_transform=use_fixed_timestep_transform,
            **kwargs,
        )

    def sample(
        self,
        model,
        y,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        
        # text encoding        
        model_args = y
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps        
        # def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
        #     if linear_steps is None:
        #         linear_steps = num_steps // 2
        #     linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        #     threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
        #     quadratic_steps = num_steps - linear_steps
        #     quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
        #     linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
        #     const = quadratic_coef * (linear_steps ** 2)
        #     quadratic_sigma_schedule = [
        #         quadratic_coef * (i ** 2) + linear_coef * i + const
        #         for i in range(linear_steps, num_steps)
        #     ]
        #     sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        #     sigma_schedule = [1.0 - x for x in sigma_schedule]
        #     return sigma_schedule

        # sigmas = linear_quadratic_schedule(self.num_sampling_steps, 0.025)
        
        # timesteps = [torch.tensor([t] * z.shape[0], device=device) * self.num_timesteps for t in sigmas[:-1]] 
        
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, scale=self.scheduler.transform_scale, num_timesteps=self.num_timesteps, fixed_timestep_transform=self.use_fixed_timestep_transform) for t in timesteps]
       
        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        dtype = z.dtype
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            pred = model(z_in.to(dtype), t.to(dtype), **model_args)
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            # for linear_quadratic_schedule
            # dt = sigmas[i] - sigmas[i + 1]
            # z = z + v_pred * dt
            
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, xt_mask=None, xs_mask=None, weights=None, t=None, mae_loss_coef=None, unpatchify_loss=False):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, xt_mask, xs_mask, weights, t, mae_loss_coef, unpatchify_loss)
