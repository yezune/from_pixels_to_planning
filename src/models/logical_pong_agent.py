import torch
import torch.nn as nn
import torch.nn.functional as F
from src.l_fep.activation import SphericalActivation

class LogicalPongAgent(nn.Module):
    """
    Logical Agent for Pong (CNN-based).
    Uses Spherical Activation for policy output.
    """
    def __init__(self, input_channels, action_dim, hidden_dim=512):
        super(LogicalPongAgent, self).__init__()
        # Input: (B, C*k, H, W) -> (B, 12, 64, 64) for k=4, C=3
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate FC input size
        # 64x64 -> conv1(s4) -> 15x15 -> conv2(s2) -> 6x6 -> conv3(s1) -> 4x4
        self.fc_input_dim = 64 * 4 * 4
        
        self.fc1 = nn.Linear(self.fc_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
        self.activation = SphericalActivation()
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Output is a vector on the hypersphere (amplitude)
        # Probabilities will be x^2
        return self.activation(x)
