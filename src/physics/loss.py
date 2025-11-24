import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_pde=0.1):
        super().__init__()
        self.lambda_pde = lambda_pde
        self.mse = nn.MSELoss()

    def forward(self, u_pred, u_true, pde_residual=None, f_true=None):
        data_loss = self.mse(u_pred, u_true)
        
        pde_loss = 0.0
        if pde_residual is not None and f_true is not None:
            pde_loss = self.mse(pde_residual, f_true)
            
        return data_loss + self.lambda_pde * pde_loss

class CausalityLoss(nn.Module):
    """
    Penalize gradients with respect to future data.
    """
    def __init__(self, manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, model, u_coeffs, query_points, input_points):
        """
        Penalize gradients outside the past light cone.
        
        Args:
            model: The GNO model
            u_coeffs: Input features (branch input) [batch, N_in]
            query_points: Output locations (trunk input) [batch, N_out, dim]
            input_points: Locations of input function u(x) [batch, N_in, dim]
                          (Note: u_coeffs usually correspond to these points or modes)
        
        Returns:
            loss: scalar
        """
        # We need gradients of output w.r.t. input u_coeffs
        # But u_coeffs are discrete values or spectral coeffs.
        # If u_coeffs are values at input_points, we can define causality.
        
        # Enable grad for input
        u_coeffs.requires_grad_(True)
        
        # Forward pass
        # pred: [batch, N_out, 1]
        pred = model(u_coeffs, query_points)
            
        gradients = torch.autograd.grad(
            outputs=pred,
            inputs=u_coeffs,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # We want to check d(pred_j)/d(u_i)
        # This is the Jacobian. Computing full Jacobian is expensive.
        # We can compute grad of sum(pred) or random projection.
        
        # Let's take a random subset of outputs to save memory
        # or just sum.
        
        grad_outputs = torch.ones_like(pred)
        gradients = torch.autograd.grad(
            outputs=pred,
            inputs=u_coeffs,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        # gradients: [batch, N_in] - this aggregates impact on ALL outputs.
        # This is not enough. We need pairwise impact.
        
        # Pairwise approach (expensive but rigorous):
        # Loop over a few query points?
        
        loss = 0.0
        batch_size = query_points.shape[0]
        num_queries = query_points.shape[1]
        num_inputs = input_points.shape[1]
        
        # Sample a few query points to check
        num_samples = min(5, num_queries)
        indices = torch.randperm(num_queries)[:num_samples]
        
        for idx in indices:
            # pred_slice: [batch, 1]
            pred_slice = pred[:, idx, 0]
            
            # Grad w.r.t u_coeffs: [batch, N_in]
            grads = torch.autograd.grad(
                outputs=pred_slice.sum(),
                inputs=u_coeffs,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Check causality for this query point
            # query_loc: [batch, dim]
            query_loc = query_points[:, idx, :]
            
            # We need to check if input_points[j] is in J-(query_loc)
            # is_causal(input, query) -> True if query is in future of input
            # We want to penalize if NOT is_causal
            
            # Expand for broadcasting
            # query_loc: [batch, 1, dim]
            q_expanded = query_loc.unsqueeze(1).expand(-1, num_inputs, -1)
            
            # input_points: [batch, N_in, dim]
            
            # Mask: 1 if CAUSAL (allowed), 0 if NOT CAUSAL (forbidden)
            # is_causal returns True if q is in future of input
            
            is_allowed = self.manifold.is_causal(input_points, q_expanded)
            
            # We want to penalize grads where is_allowed is False
            # penalty_mask = ~is_allowed
            penalty_mask = (~is_allowed).float()
            
            # Loss = sum(|grad| * penalty_mask)
            loss += torch.mean((grads * penalty_mask) ** 2)
            
        return loss
