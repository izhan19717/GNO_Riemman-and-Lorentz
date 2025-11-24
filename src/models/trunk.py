import torch
import torch.nn as nn

class TrunkNet(nn.Module):
    """
    Trunk Network: Maps geometric features to latent space.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        modules = []
        modules.append(nn.Linear(input_dim, hidden_dim))
        modules.append(nn.ReLU()) # Or Tanh/GELU for smooth functions
        
        for _ in range(layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.ReLU())
            
        modules.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, points, input_dim] or [batch, input_dim]
        """
        return self.net(x)
