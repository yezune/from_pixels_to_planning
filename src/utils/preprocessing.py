import cv2
import numpy as np
import torch

def preprocess_observation(obs, target_size=(64, 64), grayscale=True):
    """
    Preprocesses a raw observation (image) from the environment.
    
    Args:
        obs (np.ndarray): Raw observation from the environment.
        target_size (tuple): Desired output size (width, height).
        grayscale (bool): Whether to convert to grayscale.
        
    Returns:
        torch.Tensor: Preprocessed observation tensor.
    """
    if obs is None:
        return torch.zeros((1, *target_size)) if grayscale else torch.zeros((3, *target_size))

    # Ensure observation is a numpy array
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)

    # Resize
    processed = cv2.resize(obs, target_size, interpolation=cv2.INTER_AREA)

    # Convert to grayscale if requested
    if grayscale and len(processed.shape) == 3:
        processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        # Add channel dimension for grayscale: (H, W) -> (H, W, 1)
        processed = np.expand_dims(processed, axis=-1)
    elif not grayscale and len(processed.shape) == 2:
        # Convert grayscale to RGB if requested but input is gray
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

    # Normalize to [0, 1]
    processed = processed.astype(np.float32) / 255.0

    # Transpose to channel-first format (C, H, W) for PyTorch
    # Input is (H, W, C), Output is (C, H, W)
    processed = np.transpose(processed, (2, 0, 1))

    # Convert to tensor
    return torch.from_numpy(processed)

def batch_preprocess(observations, device='cpu'):
    """
    Preprocesses a batch of observations.
    """
    tensors = [preprocess_observation(obs) for obs in observations]
    return torch.stack(tensors).to(device)
