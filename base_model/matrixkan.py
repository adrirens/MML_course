import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# =================================================================================
# Section 1: Final, Robust, Batch-First Helper Functions
# =================================================================================

def B_batch(x, grid, k):
    """
    Final, robust, batch-first B-spline basis function calculator.
    Args:
        x (torch.Tensor): Input tensor, shape (batch_size, in_dim).
        grid (torch.Tensor): Grid tensor, shape (in_dim, num_grid_points).
        k (int): B-spline order.
    Returns:
        torch.Tensor: B-spline basis values, shape (batch_size, in_dim, num_coeffs).
    """
    def extend_grid_internal(grid_tensor, k_extend):
        h = (grid_tensor[:, [-1]] - grid_tensor[:, [0]]) / (grid_tensor.shape[1] - 1)
        for _ in range(k_extend):
            grid_tensor = torch.cat([grid_tensor[:, [0]] - h, grid_tensor], dim=1)
            grid_tensor = torch.cat([grid_tensor, grid_tensor[:, [-1]] + h], dim=1)
        return grid_tensor

    extended_grid = extend_grid_internal(grid, k_extend=k)
    
    x = x.unsqueeze(-1)
    grid = extended_grid.unsqueeze(0)

    bases = ((x >= grid[..., :-1]) & (x < grid[..., 1:])).to(x.dtype)
    for order in range(1, k + 1):
        denom1 = grid[..., order:-1] - grid[..., :-(order + 1)]
        denom2 = grid[..., (order + 1):] - grid[..., 1:-order]
        term1 = (x - grid[..., :-(order + 1)]) / torch.where(denom1 == 0, torch.tensor(1e-8, device=x.device), denom1) * bases[..., :-1]
        term2 = (grid[..., (order + 1):] - x) / torch.where(denom2 == 0, torch.tensor(1e-8, device=x.device), denom2) * bases[..., 1:]
        bases = term1 + term2
        
    return torch.nan_to_num(bases)

def curve2coef(x_eval, y_eval, grid, k, lamb=1e-8):
    """
    Final, correct curve-to-coefficient function using a batch-first convention.
    """
    # x_eval: (batch, in_dim), y_eval: (batch, in_dim, out_dim), grid: (in_dim, n_grid_pts)
    mat = B_batch(x_eval, grid, k) # mat shape: (batch, in_dim, n_coef)
    
    # We need to solve for each in_dim independently. Permute to (in_dim, batch, ...)
    mat_p = mat.permute(1, 0, 2)       # Shape: (in_dim, batch, n_coef)
    y_eval_p = y_eval.permute(1, 0, 2) # Shape: (in_dim, batch, out_dim)

    # Solve Ax=b where A is mat_p and b is y_eval_p.
    # The dimensions now match the requirements of lstsq.
    try:
        coef = torch.linalg.lstsq(mat_p, y_eval_p).solution # solution shape: (in_dim, n_coef, out_dim)
    except torch.linalg.LinAlgError:
        A_t_A = mat_p.transpose(-2, -1) @ mat_p
        A_t_b = mat_p.transpose(-2, -1) @ y_eval_p
        identity = torch.eye(A_t_A.shape[-1], device=A_t_A.device).unsqueeze(0)
        coef = torch.linalg.solve(A_t_A + lamb * identity, A_t_b)

    # Permute to our desired storage format: (in_dim, out_dim, n_coef)
    return coef.permute(0, 2, 1)

# =================================================================================
# Section 2: KANLinear and KAN classes using the robust helpers
# =================================================================================

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, **kwargs):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Grid initialization
        grid = torch.linspace(-1, 1, steps=grid_size + 1)[None,:].expand(self.in_features, -1)
        self.grid = nn.Parameter(grid, requires_grad=False)
        
        # Coefficient initialization (now using corrected, batch-first functions)
        x_dummy = torch.linspace(-1, 1, 100).unsqueeze(1).expand(-1, self.in_features)
        y_dummy = torch.sin(x_dummy * math.pi).unsqueeze(-1).expand(-1, -1, self.out_features)
        
        # self.coef shape will be (in_features, out_features, num_coeffs)
        self.coef = nn.Parameter(curve2coef(x_dummy, y_dummy, self.grid, self.spline_order))

        # Base function is now a simple learnable affine transformation
        self.base_weight = nn.Parameter(torch.randn(self.in_features, self.out_features))
        self.base_bias = nn.Parameter(torch.zeros(self.out_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        # x shape: (batch, in_features)
        
        # 1. Spline part
        # B_batch returns (batch, in_features, num_coeffs)
        spline_bases = B_batch(x, self.grid, self.spline_order)
        
        # self.coef shape: (in_features, out_features, num_coeffs)
        # Combine bases with coefficients
        spline_val = torch.einsum('bik,iok->bio', spline_bases, self.coef)

        # 2. Base part (a simple linear transformation)
        base_val = x @ self.base_weight + self.base_bias
        
        # 3. Summing up contributions
        # Sum spline contributions over the input dimension
        spline_sum = torch.sum(spline_val, dim=1) # shape: (batch, out_features)
        
        # The final output is the sum of the base function and summed spline activations
        output = base_val + spline_sum
        
        return output

class KAN(torch.nn.Module):
    def __init__(self, layers_hidden, **kwargs):
        super(KAN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(KANLinear(in_features, out_features, **kwargs))

    def forward(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                # Using LayerNorm is more stable than tanh for arbitrary output ranges
                x = F.layer_norm(x, x.shape[1:])
        return x

# =================================================================================
# Section 3: Example Usage
# =================================================================================

if __name__ == '__main__':
    input_features = 28 * 28
    hidden_features = 64
    output_features = 10
    layers = [input_features, hidden_features, output_features]
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    model = KAN(layers_hidden=layers, grid_size=5, spline_order=3)
    model.to(dev)
    
    dummy_input = torch.randn(128, input_features, device=dev)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model ran successfully! ✅")