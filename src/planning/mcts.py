"""
Monte Carlo Tree Search (MCTS) Planner for Latent Space Planning.

This module implements MCTS for planning in the latent space of a learned
generative model. The planner uses the transition model to simulate future
states and selects actions that minimize Expected Free Energy.
"""

import torch
import numpy as np
from typing import Optional, Tuple


class MCTSNode:
    """A node in the MCTS tree."""
    
    def __init__(self, state: torch.Tensor, parent=None, action=None):
        """
        Initialize MCTS node.
        
        Args:
            state: Latent state representation (1, latent_dim)
            parent: Parent node
            action: Action that led to this state
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.untried_actions = []
    
    def is_fully_expanded(self, action_dim: int) -> bool:
        """Check if all actions have been tried from this node."""
        return len(self.children) == action_dim
    
    def best_child(self, exploration_weight: float = 1.414) -> 'MCTSNode':
        """Select best child using UCB1 formula."""
        choices_weights = []
        for child in self.children.values():
            if child.visit_count == 0:
                weight = float('inf')
            else:
                # UCB1 formula
                exploitation = child.value_sum / child.visit_count
                exploration = exploration_weight * np.sqrt(
                    np.log(self.visit_count) / child.visit_count
                )
                weight = exploitation + exploration
            choices_weights.append(weight)
        
        return list(self.children.values())[np.argmax(choices_weights)]
    
    def most_visited_child(self) -> Tuple[int, 'MCTSNode']:
        """Return action and child with most visits."""
        visit_counts = [(action, child.visit_count) 
                       for action, child in self.children.items()]
        best_action, _ = max(visit_counts, key=lambda x: x[1])
        return int(best_action), self.children[best_action]


class MCTSPlanner:
    """
    Monte Carlo Tree Search planner for latent space planning.
    
    This planner uses the learned transition model to simulate future states
    and select actions that minimize Expected Free Energy towards a goal.
    """
    
    def __init__(
        self,
        transition_model,
        action_dim: int,
        num_simulations: int = 50,
        exploration_weight: float = 1.414,
        discount: float = 0.95,
        device: str = 'cpu'
    ):
        """
        Initialize MCTS planner.
        
        Args:
            transition_model: Learned transition model P(z_{t+1} | z_t, a)
            action_dim: Number of available actions
            num_simulations: Number of MCTS simulations per planning step
            exploration_weight: UCB1 exploration constant
            discount: Discount factor for future rewards
            device: torch device
        """
        self.transition_model = transition_model
        self.action_dim = action_dim
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.discount = discount
        self.device = device
        
        self.transition_model.to(device)
        self.transition_model.eval()
    
    def plan(
        self,
        initial_state: torch.Tensor,
        goal_state: Optional[torch.Tensor] = None,
        depth: int = 5
    ) -> int:
        """
        Plan action using MCTS.
        
        Args:
            initial_state: Current latent state (1, latent_dim)
            goal_state: Target latent state (optional)
            depth: Maximum planning depth
            
        Returns:
            best_action: Selected action (integer)
        """
        root = MCTSNode(initial_state.to(self.device))
        
        for _ in range(self.num_simulations):
            node = root
            search_depth = 0
            
            # Selection: traverse tree using UCB1
            while node.is_fully_expanded(self.action_dim) and search_depth < depth:
                node = node.best_child(self.exploration_weight)
                search_depth += 1
            
            # Expansion: add new child node
            if search_depth < depth and not node.is_fully_expanded(self.action_dim):
                node = self._expand(node)
                search_depth += 1
            
            # Simulation: rollout to estimate value
            value = self._simulate(node, goal_state, depth - search_depth)
            
            # Backpropagation: update statistics
            self._backpropagate(node, value)
        
        # Select action with most visits (most robust)
        best_action, _ = root.most_visited_child()
        return best_action
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by trying an untried action."""
        # Find an untried action
        tried_actions = set(node.children.keys())
        untried_actions = [a for a in range(self.action_dim) 
                          if a not in tried_actions]
        
        action = np.random.choice(untried_actions)
        
        # Simulate transition
        next_state = self._simulate_transition(node.state, action)
        
        # Create child node
        child = MCTSNode(next_state, parent=node, action=action)
        node.children[action] = child
        
        return child
    
    def _simulate(
        self,
        node: MCTSNode,
        goal_state: Optional[torch.Tensor],
        depth: int
    ) -> float:
        """
        Simulate random rollout from node.
        
        Args:
            node: Starting node
            goal_state: Target state (optional)
            depth: Rollout depth
            
        Returns:
            value: Estimated value (negative EFE)
        """
        current_state = node.state
        total_value = 0.0
        
        for step in range(depth):
            # Random action for simulation
            action = np.random.randint(0, self.action_dim)
            
            # Predict next state
            next_state = self._simulate_transition(current_state, action)
            
            # Calculate reward (negative distance to goal)
            if goal_state is not None:
                reward = -torch.mean((next_state - goal_state.to(self.device)) ** 2).item()
            else:
                # Without goal, prefer states with low uncertainty (entropy)
                reward = -torch.mean(next_state ** 2).item()
            
            total_value += (self.discount ** step) * reward
            current_state = next_state
        
        return total_value
    
    def _simulate_transition(
        self,
        state: torch.Tensor,
        action: int
    ) -> torch.Tensor:
        """
        Simulate state transition using learned model.
        
        Args:
            state: Current state (1, latent_dim)
            action: Action to take
            
        Returns:
            next_state: Predicted next state (1, latent_dim)
        """
        with torch.no_grad():
            action_tensor = torch.tensor([action], device=self.device)
            (next_mu, next_logvar), _ = self.transition_model(
                state.to(self.device), action_tensor, None
            )
            # Sample from predicted distribution
            std = torch.exp(0.5 * next_logvar)
            eps = torch.randn_like(std)
            next_state = next_mu + eps * std
        
        return next_state
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
