import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from src.pricers.vae import VAE


def impute_vae(model, surface, mask, steps=200, lr=0.05):
    """
    Impute missing values using the trained VAE by optimizing the latent vector z
    to minimize reconstruction error on the KNOWN points.
    """
    model.eval()

    # Initialize z randomly
    z = torch.randn((1, 8), requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    # True values where mask is False (meaning not missing)
    known_mask = ~mask
    known_values = surface[known_mask]

    for i in range(steps):
        optimizer.zero_grad()
        recon = model.decode(z).view(5, 10)

        # Loss only on known points
        loss = torch.nn.functional.mse_loss(recon[known_mask], known_values)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_recon = model.decode(z).view(5, 10)

    return final_recon.numpy()


def main():
    # Load model
    model = VAE(input_dim=50, latent_dim=8)
    try:
        model.load_state_dict(torch.load("vae_weights.pth"))
    except FileNotFoundError:
        print("Model weights not found. Run train.py first.")
        return

    # Load data
    try:
        data = torch.load("iv_surfaces.pt")
        # Just grab the first valid surface
        mask = ~torch.isnan(data.view(data.shape[0], -1)).any(dim=1)
        data = data[mask]
        surface = data[0].view(5, 10)
    except FileNotFoundError:
        print("Dataset not found.")
        return

    # Create missing mask (True means missing)
    missing_mask = torch.zeros((5, 10), dtype=torch.bool)

    # Let's remove an entire "middle" section of strikes for expiries 2 and 3
    missing_mask[1:3, 3:7] = True

    surface_incomplete = surface.clone()
    surface_incomplete[missing_mask] = float("nan")

    # 1. Linear Interpolation Baseline
    target_dte = np.array([30, 60, 90, 120, 150])
    target_moneyness = np.linspace(0.8, 1.2, 10)
    M, DTE = np.meshgrid(target_moneyness, target_dte)

    known_points = []
    known_values = []

    for i in range(5):
        for j in range(10):
            if not missing_mask[i, j]:
                known_points.append([DTE[i, j], M[i, j]])
                known_values.append(surface[i, j].item())

    known_points = np.array(known_points)
    known_values = np.array(known_values)

    # Griddata
    target_points = np.column_stack((DTE.flatten(), M.flatten()))
    linear_interp = griddata(
        known_points, known_values, target_points, method="linear"
    ).reshape(5, 10)

    # 2. VAE Interpolation
    print("Running VAE imputation...")
    vae_interp = impute_vae(model, surface, missing_mask)

    # Combine known points with interpolated points
    linear_final = surface.numpy().copy()
    linear_final[missing_mask.numpy()] = linear_interp[missing_mask.numpy()]

    vae_final = surface.numpy().copy()
    vae_final[missing_mask.numpy()] = vae_interp[missing_mask.numpy()]

    # Plotting
    fig = plt.figure(figsize=(20, 6))

    # Original
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(M, DTE, surface.numpy(), cmap="viridis")
    ax1.set_title("Original Complete Surface")

    # Linear
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(M, DTE, linear_final, cmap="viridis")
    ax2.set_title("Linear Interpolation")

    # VAE
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_surface(M, DTE, vae_final, cmap="viridis")
    ax3.set_title("VAE Interpolation")

    plt.tight_layout()
    plt.savefig("interpolation_comparison.png")
    print("Saved comparison to interpolation_comparison.png")


if __name__ == "__main__":
    main()
