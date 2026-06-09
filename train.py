import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from vae_model import VAE, vae_loss_function
import os

def main():
    if not os.path.exists("iv_surfaces.pt"):
        print("Dataset iv_surfaces.pt not found. Run data_fetcher.py first.")
        return
        
    print("Loading dataset...")
    # Load dataset
    data = torch.load("iv_surfaces.pt")
    
    # Check if we need to flatten the data
    if len(data.shape) > 2:
        # Expected shape (B, 5, 10) -> (B, 50)
        data = data.reshape(data.shape[0], -1)
        
    print(f"Dataset shape: {data.shape}")
    
    # Filter out any NaNs just in case
    mask = ~torch.isnan(data).any(dim=1)
    data = data[mask]
    print(f"Dataset shape after NaN filtering: {data.shape}")
    
    dataset = TensorDataset(data)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize VAE model
    model = VAE(input_dim=50, latent_dim=8)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 200
    beta = 0.1 # Beta-VAE parameter to prevent posterior collapse
    
    recon_losses = []
    kl_losses = []
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        for batch_idx, (x,) in enumerate(dataloader):
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(x)
            
            recon_loss, kl_loss, loss = vae_loss_function(recon_batch, x, mu, logvar, beta=beta)
            
            loss.backward()
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
            
            optimizer.step()
            
        avg_recon = train_recon / len(dataloader.dataset)
        avg_kl = train_kl / len(dataloader.dataset)
        
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Recon Loss: {avg_recon:.6f}, KL Loss: {avg_kl:.6f}')
            
    # Save model
    torch.save(model.state_dict(), "vae_weights.pth")
    print("Model saved to vae_weights.pth")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.plot(kl_losses, label='KL Divergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('VAE Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_losses.png")
    print("Loss plot saved to training_losses.png")

if __name__ == "__main__":
    main()
