import torch




class ReverseDiffusion:
    @staticmethod
    @torch.no_grad()
    def p_sample(model , x_t , t, betas ,clip_range=(-1.0, 1.0)):
        
        # Initialize alphas
        # Compute alphas and alpha_bars
        alphas = 1.0 - betas
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = torch.prod(alphas[:t+1])
        
        # Pre-calculate square roots
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        # 1. Predict noise using the model
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        epsilon_theta = model(x_t, t_tensor)

        # 2. Estimate x0 from x_t and predicted noise
        # x0_pred = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * epsilon_theta) / sqrt_alpha_bar_t

        # 3. Optional clipping
        low, high = clip_range
        x0_pred = torch.clamp(x0_pred, low, high)

        # 4. Compute epsilon used for posterior mean calculation
        epsilon_used = (x_t - sqrt_alpha_bar_t * x0_pred) / sqrt_one_minus_alpha_bar_t

        # 5. Compute the mean of the posterior q(x_{t-1} | x_t, x_0)
        mean = (1.0 / sqrt_alpha_t) * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon_used)
        
        if t > 0:
            z = torch.randn_like(x_t)
            return mean + torch.sqrt(beta_t) * z
        else:
            return mean