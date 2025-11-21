import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialRGM(nn.Module):
    """
    Spatial Renormalizing Generative Model (Hierarchical Discrete State-Space Model).
    
    Structure:
    - Level 1 (Pixels -> Features): Processes 4x4 patches.
    - Level 2 (Features -> Digit): Aggregates features to global digit identity.
    
    Latent Variables:
    - z1: Discrete latent for Level 1 (e.g., 8x8 grid of categorical variables).
    - z2: Discrete latent for Level 2 (Global digit class).
    """
    def __init__(self, input_channels=1, hidden_dim=64, latent_dim=32, num_classes=10, temp=1.0):
        super(SpatialRGM, self).__init__()
        
        self.latent_dim = latent_dim # Number of categories for z1
        self.num_classes = num_classes # Number of categories for z2 (digits)
        self.temp = temp # Gumbel-Softmax temperature
        
        # --- Level 1: Bottom-Up (Pixels -> z1) ---
        # Input: 28x28 -> 7x7 feature map
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 7x7
            nn.ReLU()
        )
        # 7x7 grid of discrete variables
        self.z1_proj = nn.Conv2d(64, latent_dim, kernel_size=1) 
        
        # --- Level 2: Bottom-Up (z1 -> z2) ---
        self.enc2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes) # Logits for global class
        )
        
        # --- Level 2: Top-Down (z2 -> z1_prior) ---
        # Predicts the expected z1 distribution given z2
        self.dec2 = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 7 * 7)
        )
        
        # --- Level 1: Top-Down (z1 -> Pixels) ---
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=1), # 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1), # 28x28
            nn.Sigmoid()
        )

    def reparameterize_gumbel(self, logits, temp=1.0):
        """
        Gumbel-Softmax sampling.
        """
        if self.training:
            return F.gumbel_softmax(logits, tau=temp, hard=False, dim=1)
        else:
            # During inference, use hard one-hot
            y_soft = F.softmax(logits, dim=1)
            index = y_soft.max(1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(1, index, 1.0)
            return y_hard

    def forward(self, x):
        # --- Bottom-Up Inference ---
        
        # 1. Encode Level 1 (Pixels -> z1 logits)
        h1 = self.enc1(x) # (B, 64, 7, 7)
        z1_logits = self.z1_proj(h1) # (B, latent_dim, 7, 7)
        
        # Sample z1 (Posterior)
        # Reshape for softmax: (B, latent_dim, 7, 7) -> (B, latent_dim, 49) -> softmax over dim 1
        z1_sample = self.reparameterize_gumbel(z1_logits, self.temp)
        
        # 2. Encode Level 2 (z1 -> z2 logits)
        # We use the sampled z1 as input to the next level
        z2_logits = self.enc2(z1_sample) # (B, num_classes)
        
        # Sample z2 (Posterior)
        z2_sample = self.reparameterize_gumbel(z2_logits, self.temp)
        
        # --- Top-Down Generation / Prediction ---
        
        # 3. Predict Level 1 Prior (z2 -> z1_prior)
        z1_prior_logits = self.dec2(z2_sample)
        z1_prior_logits = z1_prior_logits.view(-1, self.latent_dim, 7, 7)
        
        # 4. Decode Level 1 (z1 -> Pixels)
        # We use the POSTERIOR z1 for reconstruction (Autoencoder style)
        recon = self.dec1(z1_sample)
        
        # --- Loss Calculation ---
        
        # 1. Reconstruction Loss
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        
        # 2. KL Divergence for z2 (Prior is uniform 1/K)
        # KL(q(z2|z1) || p(z2))
        # p(z2) is uniform, so minimizing KL is maximizing entropy H(q)
        # But usually we want to match a specific prior. Here we just let it learn to classify.
        # We will treat z2 supervision as the main signal, so we might not strictly need KL(z2) if we have labels.
        # But for pure generative modeling, we'd want KL.
        q_z2 = F.softmax(z2_logits, dim=1)
        log_q_z2 = F.log_softmax(z2_logits, dim=1)
        # Uniform prior: log(1/K) = -log(K)
        log_p_z2 = -torch.log(torch.tensor(float(self.num_classes), device=x.device))
        kl_z2 = torch.sum(q_z2 * (log_q_z2 - log_p_z2), dim=1).sum()
        
        # 3. KL Divergence for z1 (Prior is predicted from z2)
        # KL(q(z1|x) || p(z1|z2))
        # This aligns the bottom-up posterior with the top-down prediction.
        q_z1 = F.softmax(z1_logits, dim=1) # (B, D, H, W)
        log_q_z1 = F.log_softmax(z1_logits, dim=1)
        
        # p(z1|z2) comes from dec2
        log_p_z1 = F.log_softmax(z1_prior_logits, dim=1)
        
        kl_z1 = torch.sum(q_z1 * (log_q_z1 - log_p_z1), dim=1).sum() # Sum over channel, then H, W
        
        loss_dict = {
            'recon_loss': recon_loss,
            'kl_z1': kl_z1,
            'kl_z2': kl_z2,
            'total_loss': recon_loss + kl_z1 + kl_z2,
            'logits': z2_logits # For classification loss
        }
        
        return recon, z2_logits, loss_dict

    def set_temperature(self, temp):
        """Update Gumbel-Softmax temperature."""
        self.temp = temp

    def generate(self, digit_class, device):
        """
        Generate an image for a specific digit class.
        """
        self.eval()
        with torch.no_grad():
            # Create one-hot vector for the class
            z2 = torch.zeros(1, self.num_classes, device=device)
            z2[0, digit_class] = 1.0
            
            # Top-down generation
            z1_logits = self.dec2(z2)
            z1_logits = z1_logits.view(-1, self.latent_dim, 7, 7)
            
            # Sample z1 (greedy/argmax for clean generation)
            z1 = self.reparameterize_gumbel(z1_logits)
            
            # Decode
            img = self.dec1(z1)
            return img
