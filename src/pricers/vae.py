import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Variational Autoencoder for Volatility Surface generation.

    Args:
        input_dim (int, optional): Input dimension. Defaults to 50.
        latent_dim (int, optional): Latent space dimension. Defaults to 8.
    """

    def __init__(self, input_dim=50, latent_dim=8):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)

        # Latent space
        self.fc_mean = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 32)
        self.fc4 = nn.Linear(32, 64)
        self.fc5 = nn.Linear(64, input_dim)

    def encode(self, x):
        """Encode input to latent space.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing mu and logvar tensors.
        """
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc_mean(h2), self.fc_logvar(h2)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick.

        Args:
            mu (torch.Tensor): Mean of the latent space.
            logvar (torch.Tensor): Log-variance of the latent space.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent vector to input space.

        Args:
            z (torch.Tensor): Latent vector.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        # No activation on the final layer as IV can be continuous positive values,
        # but we could use softplus or similar if we wanted strictly positive.
        # Since we might have normalized it or want exact values, we leave it linear
        # or use softplus to ensure positive volatility. Let's use Softplus.
        return F.softplus(self.fc5(h4))

    def forward(self, x):
        """Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing reconstruction, mu, logvar.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """Calculate the VAE loss.

    Args:
        recon_x (torch.Tensor): Reconstructed input.
        x (torch.Tensor): Original input.
        mu (torch.Tensor): Mean of latent space.
        logvar (torch.Tensor): Log variance of latent space.
        beta (float, optional): Beta weight for KL Divergence. Defaults to 1.0.

    Returns:
        tuple: A tuple containing MSE loss, KLD loss, and Total loss.
    """
    # Reconstruction loss (MSE)
    MSE = F.mse_loss(recon_x, x, reduction="sum")

    # KL Divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE, KLD, MSE + beta * KLD
