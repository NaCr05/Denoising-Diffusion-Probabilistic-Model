import torch

class ForwardDiffusion:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Define beta schedule (Linear schedule)
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        
        # Pre-calculate square roots for efficiency
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        
        # Schedule for noise levels 
        print("Schedule : ")
        print(">>Beta range : %f to %f" % (self.beta_start, self.beta_end))
        print(">>Alpha range : %f to %f" % (self.alphas[0].item(), self.alphas[-1].item()))
    
    def q_sample(self, x_0 , t, noise=None):
        """
        Sample from the forward diffusion process at timestep t.
        
        Parameters:
        x_0 : torch.Tensor
            The original data (clean image).
        t : int
            The timestep at which to sample.
        noise : torch.Tensor, optional
            The noise to add. If None, standard normal noise is used.
        
        Returns:
        torch.Tensor
            The noised data at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].reshape(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1)
        
        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
    
    
    def q_step(self, x_prev , t , noise=None):
        """
        Perform a single step in the forward diffusion process.
        
        Parameters:
        x_prev : torch.Tensor
            The data at the previous timestep.
        t : int
            The current timestep.
        noise : torch.Tensor, optional
            The noise to add. If None, standard normal noise is used.
        
        Returns:
        torch.Tensor
            The noised data at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_prev)
        
        beta_t = self.betas[t]
        sqrt_one_minus_beta_t = torch.sqrt(1.0 - beta_t)
        
        return sqrt_one_minus_beta_t * x_prev + torch.sqrt(beta_t) * noise
    