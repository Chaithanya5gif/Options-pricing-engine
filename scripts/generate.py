import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vae_model import VAE

def main():
    # Load model
    model = VAE(input_dim=50, latent_dim=8)
    try:
        model.load_state_dict(torch.load("vae_weights.pth"))
        model.eval()
    except FileNotFoundError:
        print("Model weights not found. Run train.py first.")
        return
        
    print("Generating 5 new implied volatility surfaces...")
    
    # Sample 5 latent vectors from N(0, I)
    z_samples = torch.randn(5, 8)
    
    with torch.no_grad():
        generated_surfaces = model.decode(z_samples)
        
    # Reshape to (5, 5, 10) -> 5 surfaces, 5 expiries, 10 strikes
    generated_surfaces = generated_surfaces.view(5, 5, 10).numpy()
    
    # Setup grid for plotting
    target_dte = np.array([30, 60, 90, 120, 150])
    target_moneyness = np.linspace(0.8, 1.2, 10)
    M, DTE = np.meshgrid(target_moneyness, target_dte)
    
    fig = plt.figure(figsize=(20, 10))
    
    for i in range(5):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        
        surface = generated_surfaces[i]
        
        # Plot surface
        surf = ax.plot_surface(M, DTE, surface, cmap='viridis', edgecolor='none')
        
        ax.set_title(f'Generated Surface {i+1}')
        ax.set_xlabel('Moneyness (K/S0)')
        ax.set_ylabel('Days to Expiry (DTE)')
        ax.set_zlabel('Implied Volatility')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
    plt.tight_layout()
    plt.savefig("generated_surfaces.png")
    print("Saved generated surfaces to generated_surfaces.png")

if __name__ == "__main__":
    main()
