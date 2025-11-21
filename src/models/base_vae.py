import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseVAE(nn.Module, ABC):
    def __init__(self):
        super(BaseVAE, self).__init__()

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, z):
        pass

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, kld_weight=1.0):
        """
        Computes the VAE loss function.
        """
        # Reconstruction loss (MSE)
        # reduction='sum' means we sum over all elements in the batch
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL Divergence
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + kld_weight * kld_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kld_loss': kld_loss
        }
