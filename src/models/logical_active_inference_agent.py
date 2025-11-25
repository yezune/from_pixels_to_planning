import torch
from src.models.agent import ActiveInferenceAgent

class LogicalActiveInferenceAgent(ActiveInferenceAgent):
    def __init__(self, rgm, transition_model, action_dim, device):
        super().__init__(rgm, transition_model, action_dim, device)
        self.rgm = rgm
        
    def infer_state(self, observation):
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        else:
            observation = observation.to(self.device)
        
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)
            
        if observation.shape[-1] in [1, 3] and observation.shape[1] not in [1, 3]:
             observation = observation.permute(0, 3, 1, 2)
             
        # Normalize if needed (assuming input is 0-255 from env)
        if observation.max() > 1.0:
            observation = observation / 255.0
            
        with torch.no_grad():
            recon, z1, z2, z1_prior = self.rgm(observation)
        
        # Use z2 (Global Concept) for planning
        return z2
