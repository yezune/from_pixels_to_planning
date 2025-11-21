"""
Trajectory Optimizer for gradient-based planning in latent space.

This module implements trajectory optimization using gradient descent
to find optimal action sequences that reach a goal state while minimizing
Expected Free Energy.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union


class TrajectoryOptimizer:
    """
    Gradient-based trajectory optimizer for latent space planning.
    
    Optimizes a sequence of actions to minimize the distance to a goal state
    while considering model uncertainty.
    """
    
    def __init__(
        self,
        transition_model,
        action_dim: int,
        horizon: int = 5,
        learning_rate: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Initialize trajectory optimizer.
        
        Args:
            transition_model: Learned transition model P(z_{t+1} | z_t, a)
            action_dim: Number of available actions
            horizon: Planning horizon (number of steps)
            learning_rate: Learning rate for gradient descent
            device: torch device
        """
        self.transition_model = transition_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.device = device
        
        self.transition_model.to(device)
        self.transition_model.eval()
    
    def optimize(
        self,
        initial_state: torch.Tensor,
        goal_state: torch.Tensor,
        num_iterations: int = 20,
        temperature: float = 1.0
    ) -> List[int]:
        """
        Optimize action sequence to reach goal state.
        
        Args:
            initial_state: Starting latent state (1, latent_dim)
            goal_state: Target latent state (1, latent_dim)
            num_iterations: Number of optimization iterations
            temperature: Temperature for softmax action selection
            
        Returns:
            action_sequence: List of discrete actions
        """
        initial_state = initial_state.to(self.device)
        goal_state = goal_state.to(self.device)
        
        # Initialize action logits (continuous relaxation of discrete actions)
        action_logits = torch.zeros(
            self.horizon, self.action_dim, 
            device=self.device, 
            requires_grad=True
        )
        
        optimizer = torch.optim.Adam([action_logits], lr=self.learning_rate)
        
        best_loss = float('inf')
        best_logits = None
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward simulation with soft actions
            current_state = initial_state
            total_loss = 0.0
            
            for t in range(self.horizon):
                # Soft action (Gumbel-Softmax relaxation)
                action_probs = F.softmax(action_logits[t] / temperature, dim=0)
                
                # Predict next states for all actions (weighted by probability)
                next_state = torch.zeros_like(current_state)
                
                for a in range(self.action_dim):
                    action_tensor = torch.tensor([a], device=self.device)
                    (next_mu, next_logvar), _ = self.transition_model(
                        current_state, action_tensor, None
                    )
                    # Weighted by action probability
                    next_state = next_state + action_probs[a] * next_mu
                
                current_state = next_state
                
                # Goal-reaching cost (distance to goal)
                goal_cost = torch.mean((current_state - goal_state) ** 2)
                
                # Entropy bonus (encourage exploration)
                entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
                
                # Combined loss (minimize distance, maximize entropy)
                step_loss = goal_cost - 0.01 * entropy
                total_loss = total_loss + step_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Track best solution
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_logits = action_logits.detach().clone()
        
        # Extract discrete actions from optimized logits
        if best_logits is not None:
            action_sequence = self._extract_discrete_actions(best_logits)
        else:
            action_sequence = self._extract_discrete_actions(action_logits.detach())
        
        return action_sequence
    
    def _extract_discrete_actions(
        self, 
        action_logits: torch.Tensor
    ) -> List[int]:
        """
        Convert continuous action logits to discrete actions.
        
        Args:
            action_logits: (horizon, action_dim) logits
            
        Returns:
            actions: List of discrete action indices
        """
        actions = []
        for t in range(self.horizon):
            action = torch.argmax(action_logits[t]).item()
            actions.append(action)
        return actions
    
    def optimize_cross_entropy(
        self,
        initial_state: torch.Tensor,
        goal_state: torch.Tensor,
        num_samples: int = 100,
        num_elite: int = 10,
        num_iterations: int = 5
    ) -> List[int]:
        """
        Alternative optimization using Cross-Entropy Method (CEM).
        
        This is a derivative-free method that samples action sequences
        and iteratively refines the distribution.
        
        Args:
            initial_state: Starting state
            goal_state: Target state
            num_samples: Number of trajectories to sample per iteration
            num_elite: Number of best trajectories to keep
            num_iterations: Number of CEM iterations
            
        Returns:
            action_sequence: Best action sequence found
        """
        initial_state = initial_state.to(self.device)
        goal_state = goal_state.to(self.device)
        
        # Initialize action distribution (uniform)
        action_probs = torch.ones(
            self.horizon, self.action_dim, device=self.device
        ) / self.action_dim
        
        best_sequence = None
        best_cost = float('inf')
        
        for iteration in range(num_iterations):
            # Sample action sequences
            samples = []
            costs = []
            
            for _ in range(num_samples):
                sequence = []
                for t in range(self.horizon):
                    action = torch.multinomial(action_probs[t], 1).item()
                    sequence.append(action)
                
                # Evaluate sequence
                cost = self._evaluate_sequence(sequence, initial_state, goal_state)
                samples.append(sequence)
                costs.append(cost)
            
            # Select elite samples
            elite_indices = np.argsort(costs)[:num_elite]
            elite_samples = [samples[i] for i in elite_indices]
            
            # Update best
            if costs[elite_indices[0]] < best_cost:
                best_cost = costs[elite_indices[0]]
                best_sequence = elite_samples[0]
            
            # Update distribution based on elites
            action_counts = torch.zeros_like(action_probs)
            for sequence in elite_samples:
                for t, action in enumerate(sequence):
                    action_counts[t, action] += 1
            
            # Smooth update
            action_probs = 0.7 * (action_counts / num_elite) + 0.3 * action_probs
            action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
        
        return best_sequence if best_sequence is not None else [0] * self.horizon
    
    def _evaluate_sequence(
        self,
        action_sequence: List[int],
        initial_state: torch.Tensor,
        goal_state: torch.Tensor
    ) -> float:
        """
        Evaluate the cost of an action sequence.
        
        Args:
            action_sequence: List of actions
            initial_state: Starting state
            goal_state: Target state
            
        Returns:
            cost: Total cost (lower is better)
        """
        current_state = initial_state
        total_cost = 0.0
        
        with torch.no_grad():
            for action in action_sequence:
                action_tensor = torch.tensor([action], device=self.device)
                (next_mu, next_logvar), _ = self.transition_model(
                    current_state, action_tensor, None
                )
                
                # Distance to goal
                cost = torch.mean((next_mu - goal_state) ** 2).item()
                total_cost += cost
                
                # Update state
                current_state = next_mu
        
        return total_cost
