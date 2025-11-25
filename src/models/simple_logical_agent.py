import torch
import torch.nn as nn
import torch.optim as optim
from src.l_fep.activation import SphericalActivation
from src.l_fep.utils import calculate_distinction_bonus

class SimpleLogicalAgent(nn.Module):
    """
    Logical Agent using L-FEP principles.
    Uses Spherical Activation for policy and Intrinsic Motivation based on Distinction.
    Refactored from src/l_fep/agent.py to separate model from library.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3, intrinsic_weight=0.01):
        super(SimpleLogicalAgent, self).__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            SphericalActivation()
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.intrinsic_weight = intrinsic_weight
        self.action_dim = action_dim
        self.saved_log_probs = []
        self.rewards = []
        self.distinctions = []

    def select_action(self, state):
        """
        Selects an action and stores log_prob and distinction.
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        psi = self.policy_net(state)
        
        # In L-FEP, probability p = psi^2 (Born rule / Spherical geometry)
        probs = psi ** 2
        
        # Calculate distinction (intrinsic reward signal)
        distinction = calculate_distinction_bonus(probs)
        self.distinctions.append(distinction)
        
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        
        return action.item()

    def update(self):
        """
        Updates the policy using REINFORCE with Intrinsic Motivation.
        """
        R = 0
        policy_loss = []
        returns = []
        
        # Combine extrinsic and intrinsic rewards
        total_rewards = []
        for r, d in zip(self.rewards, self.distinctions):
            # Intrinsic reward is the distinction bonus
            r_total = r + self.intrinsic_weight * d.item()
            total_rewards.append(r_total)
            
        for r in reversed(total_rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # Clear memory
        del self.saved_log_probs[:]
        del self.rewards[:]
        del self.distinctions[:]
        
        return loss.item()
