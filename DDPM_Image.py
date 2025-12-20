import torch
import time


from DDPM.ForwardProcess import ForwardDiffusion
from DDPM.NoisePredictor import DiffUNet
from DDPM.ReverseProcess import ReverseDiffusion
from Dataset import OxfordPetLoader

def run_reverse_process():
    
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TIMESTEPS = 1000
    N_IMAGE = 10    
    BATCH_SIZE = 10
    LR = 1e-3
    EPOCHS = 10000
    if device.type == 'cuda':
        print(f"  >> GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  >> cuDNN Version: {torch.backends.cudnn.version()}")
    
    
    # Initialize the forward and reverse diffusion processes
    forward_diffusion = ForwardDiffusion(timesteps=TIMESTEPS, beta_start=1e-4, beta_end=0.02)
    
    # Load dataset
    print(f"Loading Oxford-IIIT Pet Dataset (Cats) - Limited to {N_IMAGE} images...")
    data_loader = OxfordPetLoader(root='./data', batch_size=N_IMAGE, download=True, cat_only=True).get_loader()
    print("Dataset Loaded.")
    print("dataset size : ", len(data_loader.dataset))

    
    # Model Initialization
    model = DiffUNet(input_channels=3, time_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_function = torch.nn.MSELoss()
    
        
    

if __name__ == "__main__":
    run_reverse_process()