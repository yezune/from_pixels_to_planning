import torch
import torch.nn as nn
import torch.nn.functional as F
from src.l_fep.activation import SphericalActivation

class LogicalSpatialRGM(nn.Module):
    """
    Logical Spatial RGM (L-AGI version of SpatialRGM).
    Replaces Gumbel-Softmax with Spherical Activation (L2 Normalization).
    Latent states are represented as points on a hypersphere.
    """
    def __init__(self, input_channels=1, hidden_dim=64, latent_dim=32, num_classes=10, img_size=28):
        super(LogicalSpatialRGM, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.feature_map_size = img_size // 4
        
        # --- Level 1: Bottom-Up (Pixels -> z1) ---
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 7x7
            nn.ReLU()
        )
        # Grid of spherical variables
        self.z1_proj = nn.Conv2d(64, latent_dim, kernel_size=1)
        self.z1_act = SphericalActivation(dim=1) # Normalize across channel dim (latent_dim)
        
        # --- Level 2: Bottom-Up (z1 -> z2) ---
        self.enc2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim * self.feature_map_size * self.feature_map_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            SphericalActivation(dim=1) # Normalize across classes
        )
        
        # --- Level 2: Top-Down (z2 -> z1_prior) ---
        self.dec2 = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * self.feature_map_size * self.feature_map_size)
            # We will apply SphericalActivation after reshaping
        )
        
        # --- Level 1: Top-Down (z1 -> Pixels) ---
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Pixels are [0, 1]
        )

    def forward(self, x):
        # --- Bottom-Up Inference ---
        
        # 1. Encode Level 1 (Pixels -> z1)
        h1 = self.enc1(x)
        z1_logits = self.z1_proj(h1)
        z1 = self.z1_act(z1_logits) # (B, latent_dim, 7, 7) - Normalized
        
        # 2. Encode Level 2 (z1 -> z2)
        z2 = self.enc2(z1) # (B, num_classes) - Normalized
        
        # --- Top-Down Generation / Prediction ---
        
        # 3. Predict Level 1 Prior (z2 -> z1_prior)
        z1_prior_flat = self.dec2(z2)
        z1_prior_logits = z1_prior_flat.view(-1, self.latent_dim, self.feature_map_size, self.feature_map_size)
        z1_prior = self.z1_act(z1_prior_logits) # Normalized
        
        # 4. Decode Level 1 (z1 -> Pixels)
        recon = self.dec1(z1)
        
        # Return everything needed for loss
        return recon, z1, z2, z1_prior

    def get_loss(self, recon, x, z1, z1_prior, z2, labels=None):
        """
        Computes L-AGI Loss.
        1. Reconstruction Loss (MSE or Logical Divergence on pixels)
        2. Hierarchical Consistency (Logical Divergence between z1 and z1_prior)
        3. Classification Loss (if labels provided) - Logical Divergence on z2
        """
        # 1. Reconstruction
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        
        # 2. Hierarchical Consistency (Minimize divergence between Posterior z1 and Prior z1)
        # Logical Divergence ~ MSE on spherical vectors
        consistency_loss = F.mse_loss(z1, z1_prior, reduction='sum')
        
        # DEBUG: Disable consistency loss to check classification
        # total_loss = recon_loss + consistency_loss
        total_loss = recon_loss + consistency_loss # Re-enable consistency loss
        
        # 3. Classification (Optional)
        if labels is not None:
            # Convert labels to one-hot
            labels_one_hot = torch.zeros_like(z2)
            labels_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
            
            # L-FEP: Match probabilities (psi^2) to target probabilities
            # z2 is psi (amplitude).
            probs = z2 ** 2
            cls_loss = F.mse_loss(probs, labels_one_hot, reduction='sum')
            
            total_loss += 100.0 * cls_loss # Weight classification higher (10.0 -> 100.0)
            return total_loss, recon_loss, consistency_loss, cls_loss
            
        return total_loss, recon_loss, consistency_loss, torch.tensor(0.0)
