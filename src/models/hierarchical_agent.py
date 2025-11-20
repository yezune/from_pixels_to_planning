import torch
import torch.nn.functional as F
import numpy as np

class HierarchicalAgent:
    def __init__(self, level1_models, level2_models, action_dim, device='cpu'):
        """
        Args:
            level1_models: tuple (vae, transition_model) for the pixel level
            level2_models: tuple (vae, transition_model) for the abstract level
            action_dim: number of actions
            device: torch device
        """
        self.vae1, self.trans1 = level1_models
        self.vae2, self.trans2 = level2_models
        self.action_dim = action_dim
        self.device = device
        
        self.vae1.to(device)
        self.trans1.to(device)
        self.vae2.to(device)
        self.trans2.to(device)
        
        # Hidden states for RNNs
        self.hidden1 = None
        self.hidden2 = None
        
        self.current_z1 = None
        self.current_z2 = None

    def reset(self):
        self.hidden1 = None
        self.hidden2 = None
        self.current_z1 = None
        self.current_z2 = None

    def infer_state(self, observation):
        """
        Encodes observation to latent state z1.
        """
        observation = observation.to(self.device)
        
        # Add batch dimension if missing (C, H, W) -> (1, C, H, W)
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)
            
        with torch.no_grad():
            mu, logvar = self.vae1.encode(observation)
            z1 = self.vae1.reparameterize(mu, logvar)
        return z1

    def calculate_efe(self, next_mu, next_logvar, goal_mu=None):
        """
        Calculates Expected Free Energy.
        If goal_mu (from Level 2) is provided, it acts as the prior.
        """
        # Risk: Distance to goal (or zero if no goal)
        if goal_mu is not None:
            risk = torch.mean((next_mu - goal_mu) ** 2)
        else:
            risk = torch.mean(next_mu ** 2)
        
        # Ambiguity/Entropy: -0.5 * sum(1 + logvar)
        entropy = 0.5 * torch.sum(1 + next_logvar)
        
        efe = risk - 0.01 * entropy 
        return efe.item()

    def select_action(self, observation):
        """
        Selects an action based on minimizing Expected Free Energy.
        """
        # 1. Infer current Level 1 state
        z1_t = self.infer_state(observation)
        self.current_z1 = z1_t
        
        # 2. Update Level 2 state / Get Top-down Prior
        # For now, we assume Level 2 runs at the same tick or we just query it.
        # We pass a dummy action or the previous high-level state.
        # If current_z2 is None, initialize it (e.g. zeros)
        if self.current_z2 is None:
            # Use latent_dim from vae2 if available, otherwise infer or default
            latent_dim2 = getattr(self.vae2, 'latent_dim', 32)
            self.current_z2 = torch.zeros(1, latent_dim2, device=self.device)
            
        # Dummy action for Level 2 (if it's a transition model that needs action)
        # or maybe Level 2 just evolves autonomously.
        dummy_action = torch.zeros(1, device=self.device, dtype=torch.long) 
        
        with torch.no_grad():
            (z2_mu, z2_logvar), self.hidden2 = self.trans2(
                self.current_z2, dummy_action, self.hidden2
            )
            # Sample z2 to be the current state for next step
            self.current_z2 = self.vae2.reparameterize(z2_mu, z2_logvar)

        best_efe = float('inf')
        best_action = 0
        
        # 3. Evaluate each action (1-step lookahead)
        for a in range(self.action_dim):
            action_tensor = torch.tensor([a], device=self.device)
            
            with torch.no_grad():
                # Predict next state using Level 1 transition
                (next_mu, next_logvar), _ = self.trans1(
                    z1_t, action_tensor, self.hidden1
                )
                
                # Calculate EFE
                # Use z2_mu as the goal/prior for Level 1
                # (Assuming dimensions match or are projected)
                efe = self.calculate_efe(next_mu, next_logvar, goal_mu=z2_mu)
                
            if efe < best_efe:
                best_efe = efe
                best_action = a
        
        # 4. Update internal hidden state with the CHOSEN action
        action_tensor = torch.tensor([best_action], device=self.device)
        with torch.no_grad():
             _, self.hidden1 = self.trans1(
                z1_t, action_tensor, self.hidden1
            )
            
        return best_action
