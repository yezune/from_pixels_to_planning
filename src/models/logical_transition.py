import torch
import torch.nn as nn
from src.l_fep.activation import SphericalActivation

class LogicalTransitionModel(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.action_emb = nn.Embedding(action_dim, hidden_dim)
        self.rnn = nn.GRU(latent_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.activation = SphericalActivation()
        
    def forward(self, z, action, hidden=None):
        # z: (B, latent_dim)
        # action: (B,)
        a_emb = self.action_emb(action)
        x = torch.cat([z, a_emb], dim=1).unsqueeze(1)
        out, hidden = self.rnn(x, hidden)
        z_next = self.activation(self.fc(out.squeeze(1)))
        
        # Return dummy logvar for compatibility with MCTS/TrajectoryOptimizer
        # logvar = -20 implies std = exp(-10) ~ 4.5e-5 (very small noise)
        dummy_logvar = -20 * torch.ones_like(z_next)
        
        return (z_next, dummy_logvar), hidden
