import torch
import torch.nn.functional as F
import numpy as np

class ActiveInferenceAgent:
    def __init__(self, vae, transition_model, action_dim, device='cpu'):
        self.vae = vae
        self.transition_model = transition_model
        self.action_dim = action_dim
        self.device = device
        
        self.vae.to(device)
        self.transition_model.to(device)
        
        self.current_hidden = None
        self.current_z = None

    def reset(self):
        self.current_hidden = None
        self.current_z = None

    def infer_state(self, observation):
        """
        Encodes observation to latent state z.
        """
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        else:
            observation = observation.to(self.device)
        
        # Add batch dimension if missing (C, H, W) -> (1, C, H, W)
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)
            
        # Ensure channel-first format (B, C, H, W)
        # If input is (B, H, W, C) which is common in Gym/OpenCV, permute it.
        if observation.shape[-1] in [1, 3] and observation.shape[1] not in [1, 3]:
             observation = observation.permute(0, 3, 1, 2)
            
        with torch.no_grad():
            mu, logvar = self.vae.encode(observation)
            z = self.vae.reparameterize(mu, logvar)
        return z

    def calculate_efe(self, next_mu, next_logvar):
        """
        Calculates Expected Free Energy for a predicted state.
        Simplified version: 
        - Epistemic: Entropy of the distribution (uncertainty). We want to MAXIMIZE this for exploration? 
          Actually, in Active Inference, we minimize EFE. 
          EFE = Ambiguity + Risk.
          Ambiguity (Epistemic) is usually related to parameter uncertainty.
          Here, let's just use a simple heuristic:
          Minimize distance to a 'goal' (Risk) - Entropy (Exploration).
        
        For this implementation, we'll assume a dummy goal of z=0.
        """
        # Risk: MSE distance to zero vector (dummy goal)
        risk = torch.mean(next_mu ** 2)
        
        # Ambiguity/Entropy: -0.5 * sum(1 + logvar)
        # We want to minimize EFE, so we want to maximize Entropy.
        # So we subtract Entropy.
        entropy = 0.5 * torch.sum(1 + next_logvar)
        
        efe = risk - 0.01 * entropy # Weight exploration small
        return efe.item()

    def select_action(self, observation):
        """
        Selects an action based on minimizing Expected Free Energy.
        """
        # 1. Infer current state
        z_t = self.infer_state(observation)
        self.current_z = z_t
        
        best_efe = float('inf')
        best_action = 0
        
        # 2. Evaluate each action (1-step lookahead)
        for a in range(self.action_dim):
            action_tensor = torch.tensor([a], device=self.device)
            
            with torch.no_grad():
                # Predict next state
                (next_mu, next_logvar), _ = self.transition_model(
                    z_t, action_tensor, self.current_hidden
                )
                
                # Calculate EFE
                efe = self.calculate_efe(next_mu, next_logvar)
                
            if efe < best_efe:
                best_efe = efe
                best_action = a
        
        # 3. Update internal hidden state with the CHOSEN action
        # Note: In a real deployment, we might update the hidden state 
        # AFTER we actually take the step and get the next observation.
        # But for 'stateful' agents, we often update the RNN state here.
        # However, strictly speaking, the RNN state should track the history of (z, a).
        # So we should update it *after* we decide.
        
        action_tensor = torch.tensor([best_action], device=self.device)
        with torch.no_grad():
             _, self.current_hidden = self.transition_model(
                z_t, action_tensor, self.current_hidden
            )
            
        return best_action
