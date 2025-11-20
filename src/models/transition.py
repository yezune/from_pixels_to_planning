import torch
import torch.nn as nn

class TransitionModel(nn.Module):
    def __init__(self, latent_dim=32, action_dim=4, hidden_dim=64):
        super(TransitionModel, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Action embedding
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)

        # RNN (GRU)
        # Input size: latent_dim + hidden_dim (action embedding)
        self.rnn = nn.GRU(latent_dim + hidden_dim, hidden_dim, batch_first=True)

        # Predictor for next state (Gaussian)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_t, action, hidden=None):
        """
        Args:
            z_t: Current latent state (Batch, Latent Dim)
            action: Action indices (Batch,)
            hidden: Previous hidden state (1, Batch, Hidden Dim)
        
        Returns:
            (next_mu, next_logvar): Distribution of next latent state
            next_hidden: New hidden state
        """
        # Embed action
        action_emb = self.action_embedding(action) # (Batch, Hidden Dim)
        
        # Concatenate z_t and action
        rnn_input = torch.cat([z_t, action_emb], dim=1) # (Batch, Latent + Hidden)
        
        # Add sequence dimension for RNN: (Batch, 1, Input Size)
        rnn_input = rnn_input.unsqueeze(1)
        
        # RNN Step
        output, next_hidden = self.rnn(rnn_input, hidden)
        
        # Output is (Batch, 1, Hidden Dim) -> Squeeze to (Batch, Hidden Dim)
        output = output.squeeze(1)
        
        # Predict next state parameters
        next_mu = self.fc_mu(output)
        next_logvar = self.fc_logvar(output)
        
        return (next_mu, next_logvar), next_hidden
