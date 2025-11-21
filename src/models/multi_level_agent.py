"""
Multi-level Agent for Scale-free Active Inference.

Implements hierarchical action selection based on Expected Free Energy (EFE)
computed across all levels of the Multi-level RGM.

Key Features:
- Hierarchical state inference (bottom-up through all levels)
- Multi-level EFE calculation (considers goals at all levels)
- Temporal resolution awareness (updates levels at different rates)
- Goal-directed behavior (upper levels guide lower levels)
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np

from src.models.multi_level_rgm import MultiLevelRGM


class MultiLevelAgent:
    """
    Agent for Multi-level Hierarchical Active Inference.
    
    Selects actions by minimizing Expected Free Energy across all levels:
    - Level 0 (pixels): primitive actions
    - Level 1 (features): sub-goals
    - Level 2+ (paths): high-level goals
    
    Args:
        rgm: MultiLevelRGM model
        action_dim: Number of primitive actions
        device: torch device
    """
    
    def __init__(self, rgm: MultiLevelRGM, action_dim: int, device='cpu'):
        self.rgm = rgm
        self.action_dim = action_dim
        self.device = device
        
        # Internal state for each level
        self.hidden_states = [None] * self.rgm.num_levels
        self.current_z = [None] * self.rgm.num_levels
        
        # Timestep counter (for temporal resolution)
        self.timestep = 0
    
    def reset(self):
        """Reset agent's internal state (call at episode start)."""
        self.hidden_states = [None] * self.rgm.num_levels
        self.current_z = [None] * self.rgm.num_levels
        self.timestep = 0
    
    def infer_state(self, observation: torch.Tensor) -> List[torch.Tensor]:
        """
        Infer current state at all levels via bottom-up inference.
        
        Args:
            observation: Current observation (B, C, H, W)
            
        Returns:
            List of latent states [z0, z1, ..., z_{N-1}]
        """
        observation = observation.to(self.device)
        
        # Add batch dimension if missing
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)
        
        # Bottom-up inference through all levels
        result = self.rgm.bottom_up_inference(observation)
        
        # Update internal state
        self.current_z = result['z']
        
        return self.current_z
    
    def should_update_level(self, level: int, timestep: Optional[int] = None) -> bool:
        """
        Check if a level should update at the current timestep.
        
        Args:
            level: Level index
            timestep: Timestep to check (uses internal counter if None)
            
        Returns:
            True if level should update
        """
        if timestep is None:
            timestep = self.timestep
        
        return self.rgm.should_update_level(level, timestep)
    
    def compute_efe_single_level(
        self, 
        next_mu: torch.Tensor, 
        next_logvar: torch.Tensor,
        goal_mu: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute Expected Free Energy for a single level.
        
        EFE = Risk + Ambiguity
        
        Risk: Distance to goal (KL divergence or MSE)
        Ambiguity: Uncertainty (entropy)
        
        Args:
            next_mu: Predicted next state mean
            next_logvar: Predicted next state log variance
            goal_mu: Goal state (from upper level), None = no specific goal
            
        Returns:
            Scalar EFE value
        """
        # Risk: Distance to goal
        if goal_mu is not None:
            # Match dimensions
            if goal_mu.shape != next_mu.shape:
                # Project goal to match next_mu dimensions
                if goal_mu.shape[-1] > next_mu.shape[-1]:
                    goal_mu = goal_mu[..., :next_mu.shape[-1]]
                else:
                    padding = torch.zeros(
                        (*goal_mu.shape[:-1], next_mu.shape[-1] - goal_mu.shape[-1]),
                        device=goal_mu.device
                    )
                    goal_mu = torch.cat([goal_mu, padding], dim=-1)
            
            risk = torch.mean((next_mu - goal_mu) ** 2)
        else:
            # No goal: penalize deviation from zero (or current state)
            risk = torch.mean(next_mu ** 2)
        
        # Ambiguity: Entropy (higher is more uncertain)
        # For Gaussian: H = 0.5 * (1 + log(2π) + logvar)
        # We want to minimize uncertainty, so negative entropy
        if next_logvar is not None:
            entropy = 0.5 * torch.sum(1 + next_logvar)
            ambiguity = -entropy  # Negative because we want to reduce uncertainty
        else:
            ambiguity = 0.0
        
        # EFE = Risk (pragmatic) + β * Ambiguity (epistemic)
        # β controls exploration vs exploitation
        beta = 0.01
        efe = risk + beta * ambiguity
        
        return efe.item()
    
    def calculate_multi_level_efe(
        self, 
        action: int, 
        current_z: List[torch.Tensor]
    ) -> float:
        """
        Calculate Expected Free Energy across all levels.
        
        Each level predicts its next state and compares to the goal from the level above.
        
        Args:
            action: Primitive action to evaluate
            current_z: Current latent states at all levels
            
        Returns:
            Total EFE (sum across all levels)
        """
        total_efe = 0.0
        
        action_tensor = torch.tensor([action], device=self.device, dtype=torch.long)
        
        # Evaluate each level
        for level in range(self.rgm.num_levels):
            # Check if this level should update at current timestep
            if not self.should_update_level(level):
                continue
            
            # Predict next state at this level
            with torch.no_grad():
                next_mu, _ = self.rgm.predict_next_state(
                    level=level,
                    z_current=current_z[level],
                    action=action_tensor,
                    hidden=self.hidden_states[level]
                )
                
                # Assume some uncertainty (in practice, transition model should return logvar)
                next_logvar = torch.zeros_like(next_mu)
            
            # Get goal from upper level (if exists)
            if level + 1 < self.rgm.num_levels:
                goal_mu = current_z[level + 1].detach()
            else:
                goal_mu = None
            
            # Compute EFE for this level
            level_efe = self.compute_efe_single_level(next_mu, next_logvar, goal_mu)
            total_efe += level_efe
        
        return total_efe
    
    def select_action(self, observation: torch.Tensor) -> int:
        """
        Select action that minimizes multi-level Expected Free Energy.
        
        Process:
        1. Infer current state at all levels (bottom-up)
        2. For each action, predict next states and compute EFE
        3. Select action with minimum EFE
        4. Update internal state
        
        Args:
            observation: Current observation
            
        Returns:
            Selected action (int)
        """
        # 1. Infer current state at all levels
        current_z = self.infer_state(observation)
        
        # 2. Evaluate all actions
        best_efe = float('inf')
        best_action = 0
        
        for a in range(self.action_dim):
            efe = self.calculate_multi_level_efe(a, current_z)
            
            if efe < best_efe:
                best_efe = efe
                best_action = a
        
        # 3. Update internal state with chosen action
        action_tensor = torch.tensor([best_action], device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            for level in range(self.rgm.num_levels):
                if self.should_update_level(level):
                    _, new_hidden = self.rgm.predict_next_state(
                        level=level,
                        z_current=current_z[level],
                        action=action_tensor,
                        hidden=self.hidden_states[level]
                    )
                    self.hidden_states[level] = new_hidden
        
        # Increment timestep
        self.timestep += 1
        
        return best_action
    
    def set_goal(self, level: int, goal_z: torch.Tensor):
        """
        Manually set a goal at a specific level.
        
        This can be used for:
        - External task specification
        - Curriculum learning
        - Debugging
        
        Args:
            level: Level to set goal at
            goal_z: Goal latent state
        """
        if level < self.rgm.num_levels:
            self.current_z[level] = goal_z.to(self.device)
