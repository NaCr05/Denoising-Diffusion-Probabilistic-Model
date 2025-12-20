import torch




class ReverseDiffusion:
    @staticmethod
    @torch.no_grad()
    def p_sample(model , x_t , t, betas):
        
        # Initialize alphas
        # Compute alphas and alpha_bars
        alphas = 1.0 - betas
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = torch.prod(alphas[:t+1])
        
        # Pre-calculate square roots
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        # Noise prediction from the model
        # Make sure t is a tensor and matches the batch dimension of x_t
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        epsilon_theta = model(x_t, t_tensor)
        
        # Predict the mean of the posterior q(x_{t-1} | x_t, x_0)        
        mean = (1/sqrt_alpha_t) * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * epsilon_theta)
        
        if t > 0:
            z = torch.randn_like(x_t)
            return mean + torch.sqrt(beta_t) * z
        else:
            return mean