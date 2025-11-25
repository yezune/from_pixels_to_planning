import torch
import torch.nn as nn
import torch.nn.functional as F
from src.l_fep.activation import SphericalActivation

class LogicalCNN(nn.Module):
    """
    A Simple CNN implementing L-FEP principles (Spherical Activation).
    Based on Phase 1 of L-AGI.
    """
    def __init__(self, num_classes=10):
        super(LogicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # L-FEP Component: Spherical Activation
        self.final_activation = SphericalActivation()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Apply Spherical Activation instead of LogSoftmax
        output = self.final_activation(x)
        return output
