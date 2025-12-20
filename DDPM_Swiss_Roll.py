from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import numpy as np
import imageio
import torch

from DDPM.ForwardProcess import ForwardDiffusion
from DDPM.NoisePredictor import NoisePredictor
from DDPM.ReverseProcess import ReverseDiffusion

def run_forward_process():
    
    # Configurations
    TIMESTEPS = 200
    N_SAMPLES = 40
    
    # Initialize the forward diffusion process
    forward_diffusion = ForwardDiffusion(timesteps=TIMESTEPS, beta_start=1e-4, beta_end=0.02)
    
    # Generate Swiss Roll data
    # 40 samples with some noise
    data, _ = make_swiss_roll(n_samples=N_SAMPLES, noise=0.1)
    data = data[:, [0, 2]] # Convert 3D to 2D (X, Z)
    
    # Normalize data to [-1, 1]
    data = data - data.mean(axis=0)
    data = data / data.std(axis=0)
    x_cur = torch.tensor(data, dtype=torch.float32)
    
    # Store frames for visualization
    frames = []
    history = [x_cur.numpy().copy()]

    
    # Run the forward diffusion process
    for t in range(TIMESTEPS):
        fig, ax = plt.subplots(figsize=(6, 6))
        # Plot Trajectories
        hist_np = np.array(history)
        for p_idx in range(N_SAMPLES):
                ax.plot(hist_np[:, p_idx, 0], hist_np[:, p_idx, 1], 
                        c='gray', alpha=0.3, linewidth=1)

        ax.scatter(x_cur[:, 0], x_cur[:, 1], c='blue', alpha=0.8, s=20, zorder=5)
        ax.set_title(f"Forward Diffusion Process - Step {t}")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(f"Forward Process: t = {t}/{TIMESTEPS}")
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = image[:, :, :3]  # Convert RGBA to RGB
        frames.append(image)
        # Add pause at the beginning (t=0)
        if t == 0:
            # Repeat frame 15 times (~1.5 seconds at 10fps)
            for _ in range(15):
                frames.append(image)
        plt.close()
        
        # q_step to get next noised data
        x_cur = forward_diffusion.q_step(x_cur, t)
        # Append current state to history
        history.append(x_cur.numpy().copy())
    
    # Add pause at the end
    last_frame = frames[-1]
    for _ in range(15):
        frames.append(last_frame)
    imageio.mimsave('Plot/ddpm_forward_traj.gif', frames, fps=10, loop=0)
    print("Saved Plot/ddpm_forward_traj.gif")
    
    
def run_reverse_process():
    
    # Hyperparameters
    TIMESTEPS = 200
    BATCH_SIZE = 40
    LR = 1e-3
    EPOCHS = 20000
    N_SAMPLES = 3000
    
    
    # Training data: Swiss Roll
    data, _ = make_swiss_roll(n_samples=N_SAMPLES, noise=0.1)
    data = data[:, [0, 2]] # Convert 3D to 2D (X, Z)    
    
    # Normalize data to [-1, 1]
    data = data - data.mean(axis=0)
    data = data / data.std(axis=0)
    dataset = torch.tensor(data, dtype=torch.float32)
    
    # Initialize Forward Diffusion Process
    forward_diffusion = ForwardDiffusion(timesteps=TIMESTEPS, beta_start=1e-4, beta_end=0.02)
    model = NoisePredictor(input_dim=2, time_dim=32)
    
    # Training the Noise Predictor Model
    model.fit(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, forward_diffusion=forward_diffusion)
    
    # Visualize Reverse Diffusion Process
    frames = []
    N_GEN_SAMPLES = 1000
    
    # Start from pure noise
    x_cur = torch.randn(N_GEN_SAMPLES, 2)

    with torch.no_grad():
        for t in reversed(range(TIMESTEPS)):

            x_cur = ReverseDiffusion.p_sample(model, x_cur, t, forward_diffusion.betas)
            if t % 5 != 0:
                continue
            
            # Plotting
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(x_cur[:, 0], x_cur[:, 1], c='blue', alpha=0.8, s=20, zorder=5)
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_title(f"Reverse Process: t = {t}/{TIMESTEPS}")
            ax.grid(True, linestyle='--', alpha=0.3)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frames.append(image[:, :, :3])  # Convert RGBA to RGB
            plt.close()
            
    # Add pause at the end
    last_frame = frames[-1]
    for _ in range(15):
        frames.append(last_frame)
    imageio.mimsave('Plot/ddpm_reverse_process.gif', frames, fps=10, loop=0)
    print("Saved Plot/ddpm_reverse_process.gif")

    pass
if __name__ == "__main__":

    run_forward_process()
    run_reverse_process()