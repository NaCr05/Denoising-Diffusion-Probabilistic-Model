import torch
import torch.nn as nn
import torch.optim as optim

from DDPM.ForwardProcess import ForwardDiffusion

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class NoisePredictor(nn.Module):
    # input_dim: Dimension of the input data (e.g., 2 for 2D data)
    # time_dim: Dimension of the time embedding
    def __init__(self, input_dim = 2, time_dim = 32):
        super().__init__()
        self.time_embedding = SinusoidalPositionEmbeddings(time_dim)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim + time_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        x_and_t = torch.cat([x, t_emb], dim=-1)
        return self.network(x_and_t)
    
    def fit(self, dataset, epochs, batch_size , lr , forward_diffusion : ForwardDiffusion):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_function = nn.MSELoss()
        timesteps = forward_diffusion.timesteps
        
        print(f"Start Training ({epochs} steps)...")
        
        for epoch in range(epochs):
            # 1. Sample a batch of data
            index = torch.randint(0, len(dataset), (batch_size,))
            x_0 = dataset[index] # [batch_size, input_dim]

            # 2. Sample random timesteps 
            t = torch.randint(0, timesteps, (batch_size,), device=x_0.device)

            # 3. Forward diffusion process
            noise = torch.randn_like(x_0)
            x_t = forward_diffusion.q_sample(x_0, t , noise=noise)

            # 4. Predict noise
            predicted_noise = self.forward(x_t, t)
            
            # 5. Compute loss
            loss = loss_function(predicted_noise, noise)

            # 6. Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 2000 == 0:
                print(f"Epoch {epoch} / {epochs} : Loss = {loss.item():.6f}")
        print("Training Completed.")
        
# ---------------------------------
# DiffUNet Model (Optional - Not used in current implementation)
# ---------------------------------
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        # GroupNorm for better training stability
        # Using 32 groups for GroupNorm
        self.gn1 = nn.GroupNorm(32, out_channels)
        
        # Project time embeddings to match output channel dimension for residual connection
        self.time_mlp = nn.Linear(time_dim, out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(32, out_channels)
        
        
    def forward(self, x, t):
        # 1. Conv + GN + ReLU
        x = self.conv1
        x = self.gn1(x)
        x = torch.relu(x)
        
        # 2. Add time embedding
        t_emb = self.time_mlp(t)
        
        # Expand t_emb to match x's dimensions
        t_emb = t_emb.unsqueeze(-1)  # [batch_size, out_channels, 1]
        x = x + t_emb
        
        # 3. Conv + GN + ReLU
        x = self.conv2(x)
        x = self.gn2(x)
        x = torch.relu(x)
        return x
    
class DiffUNet(nn.Module):
    def __init__(self, input_channels=3, time_dim=32):
        super().__init__()
        
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Encoder 
        # 256 -> 128
        self.encoder1 = Block(input_channels, 64, time_dim)
        self.pool1 = nn.MaxPool1d(2)
        
        # 128 -> 64
        self.encoder2 = Block(64, 128, time_dim)
        self.pool2 = nn.MaxPool1d(2)
        
        # 64 -> 32
        self.encoder3 = Block(128, 256, time_dim)
        self.pool3 = nn.MaxPool1d(2)
        
        # 32 -> 16 
        self.encoder4 = Block(256, 512, time_dim)
        self.pool4 = nn.MaxPool1d(2)
        
        # 16 -> 8
        self.encoder5 = Block(512, 512, time_dim)
        self.pool5 = nn.MaxPool1d(2)
        
        # Bottleneck
        # feature map : 512x16x16 -> 1024x16x16
        self.bottleneck = Block(512, 1024, time_dim)
            
        # Decoder
        # 8 -> 16
        self.up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # concat with encoder5 output : 512 + 512 = 1024
        self.decoder5 = Block(1024, 512, time_dim)
        
        # 16 -> 32
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # concat with encoder4 output : 512 + 512 = 1024
        self.decoder4 = Block(1024, 512, time_dim)
        
        # 32 -> 64
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = Block(512, 256, time_dim)
        
        # 64 -> 128
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # concat with encoder2 output : 128 + 128 = 256
        self.decoder2 = Block(256, 128, time_dim)
        
        # 128 -> 256
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # concat with encoder1 output : 64 + 64 = 128
        self.decoder1 = Block(128, 64, time_dim)
        
        # Output layer
        # feature map : 64x16x16 -> 3x16x16
        self.out_conv = nn.Conv2d(64, input_channels, kernel_size=1)
        
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embedding(t)
        
        # Encoder
        e1 = self.encoder1(x, t_emb)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1, t_emb)
        p2 = self.pool2(e2)
        e3 = self.encoder3(p2, t_emb)
        p3 = self.pool3(e3)
        e4 = self.encoder4(p3, t_emb)
        p4 = self.pool4(e4)
        e5 = self.encoder5(p4, t_emb)
        p5 = self.pool5(e5)
        # Bottleneck
        b = self.bottleneck(p4, t_emb)
        # Decoder
        u5 = self.up5(b)
        d5 = self.decoder5(torch.cat([u5, e5], dim=1), t_emb)
        u4 = self.up4(b)
        d4 = self.decoder4(torch.cat([u4, e4], dim=1), t_emb)
        u3 = self.up3(d4)
        d3 = self.decoder3(torch.cat([u3, e3], dim=1), t_emb)
        u2 = self.up2(d3)
        d2 = self.decoder2(torch.cat([u2, e2], dim=1), t_emb)
        u1 = self.up1(d2)
        d1 = self.decoder1(torch.cat([u1, e1], dim=1), t_emb)
        out = self.out_conv(d1)
        return out