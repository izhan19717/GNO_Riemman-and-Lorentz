import torch
import torch.nn as nn
from .branch import BranchNet
from .trunk import TrunkNet

class GeometricDeepONet(nn.Module):
    """
    Geometric DeepONet: G(u)(y) = <B(u), T(y)>
    """
    def __init__(self, branch: BranchNet, trunk: TrunkNet):
        super().__init__()
        self.branch = branch
        self.trunk = trunk
        assert branch.output_dim == trunk.output_dim, "Branch and Trunk output dims must match"

    def forward(self, u_coeffs: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:
        """
        u_coeffs: [batch, branch_input_dim]
        y_features: [batch, num_points, trunk_input_dim]
        
        Returns: [batch, num_points, 1]
        """
        # B(u): [batch, p]
        b_out = self.branch(u_coeffs)
        
        # T(y): [batch, num_points, p]
        t_out = self.trunk(y_features)
        
        # Dot product
        # b_out needs to be broadcasted to [batch, 1, p]
        b_out_expanded = b_out.unsqueeze(1)
        
        # sum(b * t, dim=-1) -> [batch, num_points]
        res = torch.sum(b_out_expanded * t_out, dim=-1, keepdim=True)
        
        return res
