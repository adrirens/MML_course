import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import math

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.01,
        scale_base=1.0, 
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02, 
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.spline_dim = grid_size + spline_order # G+k

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, self.spline_dim) # Use self.spline_dim
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        else:
             self.register_buffer('spline_scaler', None)

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline

        self.base_activation = base_activation
        self.grid_eps = grid_eps 
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.base_weight, -1, 1)
        self.base_weight.data *= self.scale_base / math.sqrt(self.in_features)

        with torch.no_grad():
            noise_std = self.scale_noise / math.sqrt(self.in_features) / math.sqrt(self.spline_dim)
            nn.init.normal_(self.spline_weight, mean=0.0, std=noise_std)

            if self.enable_standalone_scale_spline and self.spline_scaler is not None:
                nn.init.normal_(self.spline_scaler, mean=self.scale_spline, std=0.1 * self.scale_spline)
            else:
                 if self.scale_spline != 1.0:
                      self.spline_weight.data *= self.scale_spline


    def b_splines(self, x: torch.Tensor):
        # x: (N, in_features)
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        epsilon = 1e-8
        for k in range(1, self.spline_order + 1):
             delta_prev = grid[:, k:-1] - grid[:, : -(k + 1)]
             delta_next = grid[:, k + 1 :] - grid[:, 1:(-k)]
             term1 = (x - grid[:, : -(k + 1)].unsqueeze(0)) / (delta_prev.unsqueeze(0) + epsilon) * bases[:, :, :-1]
             term2 = (grid[:, k + 1 :].unsqueeze(0) - x) / (delta_next.unsqueeze(0) + epsilon) * bases[:, :, 1:]
             bases = term1 + term2
        assert bases.size() == (x.size(0), self.in_features, self.spline_dim), f"B-spline shape mismatch: {bases.shape}"
        return bases.contiguous()

    @property
    def scaled_spline_weight(self):
        if self.enable_standalone_scale_spline and self.spline_scaler is not None:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        else:
            return self.spline_weight

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x_reshaped = x.view(-1, self.in_features) # (N, i)


        base_output = F.linear(self.base_activation(x_reshaped), self.base_weight) # (N, o)


        spline_basis = self.b_splines(x_reshaped) # (N, i, k) 
        scaled_weight = self.scaled_spline_weight # (o, i, k)
        spline_output = torch.einsum('nik,oik->no', spline_basis, scaled_weight) # (N, o)

        output = base_output + spline_output

 
        target_shape = list(original_shape[:-1]) + [self.out_features]
        output = output.reshape(target_shape)
        return output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
         l1_term = torch.mean(torch.abs(self.scaled_spline_weight))
         regularization_loss_activation = l1_term
         regularization_loss_entropy = torch.tensor(0.0, device=self.scaled_spline_weight.device)
         return (
              regularize_activation * regularization_loss_activation
              + regularize_entropy * regularization_loss_entropy
          )

class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden, 
        grid_size=5,
        spline_order=3,
        scale_noise=0.01,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        enable_standalone_scale_spline=True,
    ):
        super(KAN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    enable_standalone_scale_spline=enable_standalone_scale_spline,
                )
            )


    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        total_reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
             if hasattr(layer, 'regularization_loss') and callable(layer.regularization_loss):
                total_reg_loss += layer.regularization_loss(regularize_activation, regularize_entropy)
        return total_reg_loss
