import matplotlib.image as img
import torch
import time
import imageio
import numpy as np

from DDPM.ForwardProcess import ForwardDiffusion
from DDPM.NoisePredictor import DiffUNet
from DDPM.ReverseProcess import ReverseDiffusion
from Dataset import OxfordPetLoader


def tensor_to_image(x_tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Convert a batch tensor to uint8 images undoing normalization.

    Args:
        x_tensor: torch.Tensor, shape [B, C, H, W], model outputs in normalized space.
        mean, std: tuples for per-channel Normalize used in Dataset (default maps [-1,1]<->[0,1]).

    Returns:
        numpy array of uint8 images with shape [B, H, W, C]
    """
    # Ensure tensor on CPU and float
    x = x_tensor.detach().cpu().float()

    # If dataset used Normalize(mean, std) after ToTensor, undo it here:
    # x_orig = x * std + mean  (applied per-channel)
    mean = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)

    x = x * std + mean

    # Clip to [0,1] for safety, then convert to [0,255]
    x = torch.clamp(x, 0.0, 1.0)

    x_np = x.numpy()
    x_np = np.transpose(x_np, (0, 2, 3, 1))  # [B, H, W, C]
    x_np = (x_np * 255.0).round().astype(np.uint8)
    return x_np


def run_reverse_process():
    
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TIMESTEPS = 1000
    N_IMAGE = 5    
    BATCH_SIZE = 5
    LR = 1e-3
    EPOCHS = 20000
    if device.type == 'cuda':
        print(f"  >> GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  >> cuDNN Version: {torch.backends.cudnn.version()}")
    
    
    # Initialize the forward and reverse diffusion processes
    forward_diffusion = ForwardDiffusion(timesteps=TIMESTEPS, beta_start=1e-4, beta_end=0.02)
    # Change device of forward diffusion tensors
    forward_diffusion.to(device)
    
    # Load dataset
    print(f"Loading Oxford-IIIT Pet Dataset (Cats) - Limited to {N_IMAGE} images...")
    data_loader = OxfordPetLoader(root='./data', batch_size=N_IMAGE, download=True, cat_only=True).get_loader()
    
    # Fetch images for training
    imgs, _ = next(iter(data_loader))
    imgs = imgs[:N_IMAGE].to(device)
    print(f"Dataset Loaded. Image shape: {imgs.shape}")
    
    # Model Initialization
    model = DiffUNet(input_channels=3, time_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_function = torch.nn.MSELoss()
    
    # Training Loop
    print(f"Start Training ({EPOCHS} steps)...")
    for epoch in range(EPOCHS):
        time_start = time.time()
        total_loss = 0.0
        
        # 1. Sample random timesteps
        t = torch.randint(0, TIMESTEPS, (BATCH_SIZE,), device=device)
            
        # 2. Forward diffusion process
        noise = torch.randn_like(imgs)
        x_t = forward_diffusion.q_sample(imgs, t, noise=noise)

        # 3. Predict noise
        predicted_noise = model(x_t, t)
            
        # 4. Compute loss
        loss = loss_function(predicted_noise, noise)
            
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / N_IMAGE
        time_end = time.time()
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}, Time: {time_end - time_start:.2f}s")

            
    print("Training Completed.")
    
    # Visualize Reverse Diffusion Process
    print("Generating reverse diffusion process...")
    model.eval()
    frames = []
    
    with torch.no_grad():
        # Start from pure noise
        x_cur = torch.randn(N_IMAGE, 3, 256, 256).to(device)
        
        for t in reversed(range(TIMESTEPS)):
            # Perform one reverse diffusion step
            x_cur = ReverseDiffusion.p_sample(model, x_cur, t, forward_diffusion.betas)
            
            # Save frame every 10 timesteps
            if t % 10 == 0:
                x_images = tensor_to_image(x_cur)
                # Stack all images horizontally (or in a grid for better visualization)
                # For simplicity, we'll use the first image from the batch
                frames.append(x_images[0])
        
        # Save as GIF
        imageio.mimsave('Plot/ddpm_cat_reverse_process.gif', frames, fps=10, loop=0)
        print("Saved Plot/ddpm_cat_reverse_process.gif")



if __name__ == "__main__":
    run_reverse_process()