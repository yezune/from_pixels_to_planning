import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_shape=(1, 64, 64), latent_dim=32):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.c, self.h, self.w = input_shape

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.c, 32, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 4x4
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size
        self.flatten_size = 256 * (self.h // 16) * (self.w // 16)
        
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, self.h // 16, self.w // 16)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.c, kernel_size=4, stride=2, padding=1), # 64x64
            nn.Sigmoid() # Output between 0 and 1
        )

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        return result

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, kld_weight=1.0):
        # Reconstruction loss (MSE or BCE)
        # Using MSE here as inputs are normalized [0, 1] but not strictly binary
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL Divergence
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + kld_weight * kld_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kld_loss': kld_loss
        }
