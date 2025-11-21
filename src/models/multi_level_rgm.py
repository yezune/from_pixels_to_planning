"""
Multi-level Hierarchical Renormalizing Generative Model (RGM).

Implements the core architecture from "From pixels to planning: Scale-free active inference"
with support for N-level hierarchical structure (3+ levels recommended).

Each level consists of:
- VAE (Variational Autoencoder) for encoding/decoding
- Transition Model for temporal dynamics
- Temporal resolution (τ) for scale-free behavior

Implements:
- Bottom-up inference (observation → latents at all levels)
- Top-down prediction (upper level prior → lower level prior)
- Hierarchical loss computation across all levels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.models.base_vae import BaseVAE
from src.models.transition import TransitionModel


@dataclass
class LevelConfig:
    """Configuration for a single level in the hierarchy."""
    latent_dim: int
    temporal_resolution: int  # τ: how often this level updates (1=every step, 4=every 4 steps, etc.)
    vae: BaseVAE
    transition: TransitionModel


class MultiLevelRGM(nn.Module):
    """
    N-Level Hierarchical Renormalizing Generative Model.
    
    Implements scale-free active inference with:
    - Bottom-up inference through all levels
    - Top-down prediction from upper to lower levels
    - Hierarchical loss computation
    - Temporal abstraction via different temporal resolutions per level
    
    Args:
        level_configs: List of LevelConfig, ordered from Level 0 (pixels) to Level N-1 (highest abstraction)
        device: torch device
    """
    
    def __init__(self, level_configs: List[LevelConfig], device='cpu'):
        super().__init__()
        
        self.num_levels = len(level_configs)
        self.levels = level_configs
        self.device = device
        
        # Move all models to device
        for level in self.levels:
            level.vae.to(device)
            level.transition.to(device)
        
        # Track timestep for temporal resolution scheduling
        self.timestep = 0
    
    def reset_timestep(self):
        """Reset internal timestep counter (call at episode start)."""
        self.timestep = 0
    
    def should_update_level(self, level: int, timestep: Optional[int] = None) -> bool:
        """
        Determine if a level should be updated at the given timestep based on its temporal resolution.
        
        Args:
            level: Level index (0=lowest, num_levels-1=highest)
            timestep: Current timestep (uses internal counter if None)
            
        Returns:
            True if the level should update at this timestep
        """
        if timestep is None:
            timestep = self.timestep
        
        tau = self.levels[level].temporal_resolution
        return (timestep % tau) == 0
    
    def bottom_up_inference(self, observation: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Bottom-up inference: encode observation through all levels.
        
        Flow:
        observation (pixels) → z0 (pixel latent)
        z0 → z1 (feature latent)
        z1 → z2 (path latent)
        ...
        
        Args:
            observation: Batch of observations, shape (B, C, H,W) or (B, observation_dim)
            
        Returns:
            Dict with:
                'z': List of latent states [z0, z1, ..., z_{N-1}]
                'mu': List of means [mu0, mu1, ..., mu_{N-1}]
                'logvar': List of log variances [logvar0, logvar1, ..., logvar_{N-1}]
        """
        z_list = []
        mu_list = []
        logvar_list = []
        
        # Level 0: Encode observation
        mu_0, logvar_0 = self.levels[0].vae.encode(observation)
        z_0 = self.levels[0].vae.reparameterize(mu_0, logvar_0)
        
        z_list.append(z_0)
        mu_list.append(mu_0)
        logvar_list.append(logvar_0)
        
        # Higher levels: Encode lower level's latent
        for i in range(1, self.num_levels):
            mu_i, logvar_i = self.levels[i].vae.encode(z_list[i-1].detach())
            z_i = self.levels[i].vae.reparameterize(mu_i, logvar_i)
            
            z_list.append(z_i)
            mu_list.append(mu_i)
            logvar_list.append(logvar_i)
        
        return {
            'z': z_list,
            'mu': mu_list,
            'logvar': logvar_list
        }
    
    def top_down_prediction(self, z_upper: torch.Tensor, target_level: int) -> torch.Tensor:
        """
        Top-down prediction: generate prior for lower level from upper level's latent.
        
        Args:
            z_upper: Latent state from upper level (level = target_level + 1)
            target_level: Target level to generate prior for
            
        Returns:
            Prior (mean) for the target level, shape (B, latent_dim_target)
        """
        # Decode upper level's latent to generate prior for target level
        # In practice, this might involve a projection layer or the VAE decoder
        # For simplicity, we use the decoder to map to the target dimension
        
        upper_level_idx = target_level + 1
        target_latent_dim = self.levels[target_level].latent_dim
        
        # The upper level "predicts" what the lower level should be
        # This is typically done via the decoder of the upper level's VAE
        prior_mu = self.levels[upper_level_idx].vae.decode(z_upper)
        
        # If dimensions don't match, we need a projection
        # (In a full implementation, you'd add projection layers)
        # For now, assume dimensions are compatible or use a simple projection
        if prior_mu.shape[-1] != target_latent_dim:
            # Simple linear projection (should be learned parameter in practice)
            # For testing purposes, we just truncate or pad
            if prior_mu.shape[-1] > target_latent_dim:
                prior_mu = prior_mu[..., :target_latent_dim]
            else:
                padding = torch.zeros((*prior_mu.shape[:-1], target_latent_dim - prior_mu.shape[-1]), 
                                     device=prior_mu.device)
                prior_mu = torch.cat([prior_mu, padding], dim=-1)
        
        return prior_mu
    
    def compute_hierarchical_loss(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical loss across all levels.
        
        Loss components:
        - Level 0: Reconstruction loss (pixels) + KL divergence (uniform prior)
        - Level i (i>0): KL divergence with prior from level i+1
        
        Args:
            observation: Batch of observations
            
        Returns:
            Dictionary with individual losses and total_loss
        """
        # Bottom-up inference
        result = self.bottom_up_inference(observation)
        z_list = result['z']
        mu_list = result['mu']
        logvar_list = result['logvar']
        
        loss_dict = {}
        total_loss = 0.0
        
        # Level 0: Reconstruction + KL with uniform prior
        recon = self.levels[0].vae.decode(z_list[0])
        recon_loss = F.mse_loss(recon, observation, reduction='sum') / observation.size(0)
        
        # KL divergence: KL(q(z0|x) || p(z0)) where p(z0) is standard normal
        kl_loss_0 = -0.5 * torch.sum(1 + logvar_list[0] - mu_list[0].pow(2) - logvar_list[0].exp())
        kl_loss_0 = kl_loss_0 / observation.size(0)
        
        loss_dict['recon_loss_0'] = recon_loss
        loss_dict['kl_loss_0'] = kl_loss_0
        total_loss = total_loss + recon_loss + kl_loss_0
        
        # Higher levels: KL with prior from upper level
        for i in range(1, self.num_levels):
            # Get prior from upper level (if exists)
            if i < self.num_levels - 1:
                # Prior from level i+1
                prior_mu = self.top_down_prediction(z_list[i+1].detach(), target_level=i)
                prior_logvar = torch.zeros_like(prior_mu)  # Assume fixed variance for simplicity
                
                # KL(q(zi|z_{i-1}) || p(zi|z_{i+1}))
                kl_loss_i = 0.5 * torch.sum(
                    prior_logvar - logvar_list[i] + 
                    (logvar_list[i].exp() + (mu_list[i] - prior_mu).pow(2)) / prior_logvar.exp() - 1
                )
            else:
                # Highest level: KL with uniform prior
                kl_loss_i = -0.5 * torch.sum(1 + logvar_list[i] - mu_list[i].pow(2) - logvar_list[i].exp())
            
            kl_loss_i = kl_loss_i / observation.size(0)
            loss_dict[f'kl_loss_{i}'] = kl_loss_i
            total_loss = total_loss + kl_loss_i
        
        loss_dict['total_loss'] = total_loss
        return loss_dict
    
    def forward(self, observation: torch.Tensor) -> Dict:
        """
        Full forward pass: bottom-up inference + top-down generation + loss computation.
        
        Args:
            observation: Batch of observations
            
        Returns:
            Dictionary with latents, reconstruction, and losses
        """
        # Bottom-up
        inference_result = self.bottom_up_inference(observation)
        
        # Reconstruct observation from Level 0 latent
        recon = self.levels[0].vae.decode(inference_result['z'][0])
        
        # Compute losses
        loss_dict = self.compute_hierarchical_loss(observation)
        
        return {
            'z': inference_result['z'],
            'mu': inference_result['mu'],
            'logvar': inference_result['logvar'],
            'recon': recon,
            'losses': loss_dict
        }
    
    def predict_next_state(
        self, 
        level: int, 
        z_current: torch.Tensor, 
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict next state using the transition model at a specific level.
        
        Args:
            level: Level index
            z_current: Current latent state at this level
            action: Action taken
            hidden: Hidden state for RNN (if applicable)
            
        Returns:
            (next_z_mu, next_hidden)
        """
        (next_mu, next_logvar), next_hidden = self.levels[level].transition(
            z_current, action, hidden
        )
        return next_mu, next_hidden
