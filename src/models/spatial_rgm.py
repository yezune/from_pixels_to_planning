import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialRGM(nn.Module):
    """
    Spatial Renormalizing Generative Model (Simplified for MNIST).
    
    Structure:
    - Encoder: Compresses 28x28 image to latent vector z.
    - Classifier: Predicts class y from z (or part of the network).
    - Decoder: Reconstructs image from z.
    
    This implements a basic form of spatial abstraction where pixels are mapped to 
    abstract features (z) and then to high-level concepts (digits).
    """
    def __init__(self, input_channels=1, hidden_dim=64, latent_dim=32, num_classes=10):
        super(SpatialRGM, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder (Bottom-up)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU()
        )
        
        # Variational parameters for z
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Classifier (Latent -> Class)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Decoder (Top-down)
        self.decoder_input = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1), # 28x28
            nn.Sigmoid() # Pixel values 0-1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # Classify
        logits = self.classifier(z)
        
        # Decode
        d_in = self.decoder_input(z)
        d_in = d_in.view(-1, 64, 7, 7)
        recon = self.decoder(d_in)
        
        # Calculate Losses
        # 1. Reconstruction Loss (MSE or BCE)
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        
        # 2. KL Divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total VAE Loss (Classification loss will be added in the training loop usually, 
        # but we can return components here)
        
        loss_dict = {
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'total_loss': recon_loss + kl_loss # Note: Classification loss needs labels, so calculated outside or passed in
        }
        
        return recon, logits, loss_dict
