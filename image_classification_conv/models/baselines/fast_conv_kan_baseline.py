import torch
import torch.nn as nn

from kan_convs import FastKANConv2DLayer

from utils import L1
from typing import List

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            use_base_update: bool = True,
            base_activation=nn.SiLU,
            spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation()
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

class FastKAN(nn.Module):
    def __init__(
            self,
            layers_hidden: List[int],
            dropout: float = 0.0,
            l1_decay: float = 0.0,
            grid_range: List[float] = [-2, 2],
            grid_size: int = 8,
            use_base_update: bool = True,
            base_activation=nn.SiLU,
            spline_weight_init_scale: float = 0.1,
            first_dropout: bool = True, **kwargs
    ) -> None:
        super().__init__()
        self.layers_hidden = layers_hidden
        self.grid_min = grid_range[0]
        self.grid_max = grid_range[1]
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.spline_weight_init_scale = spline_weight_init_scale
        self.num_layers = len(layers_hidden[:-1])

        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            layer = FastKANLayer(in_features, out_features,
                                 grid_min=self.grid_min,
                                 grid_max=self.grid_max,
                                 num_grids=grid_size,
                                 use_base_update=use_base_update,
                                 base_activation=base_activation,
                                 spline_weight_init_scale=spline_weight_init_scale)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SimpleFastConvKAN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            grid_size: int = 8,
            degree_out: int = 2,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(SimpleFastConvKAN, self).__init__()

        self.layers = nn.Sequential(
            FastKANConv2DLayer(input_channels, layer_sizes[0], grid_size=grid_size, kernel_size=3, groups=1, padding=1,
                               stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            L1(FastKANConv2DLayer(layer_sizes[0], layer_sizes[1], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), l1_penalty),
            L1(FastKANConv2DLayer(layer_sizes[1], layer_sizes[2], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), l1_penalty),
            L1(FastKANConv2DLayer(layer_sizes[2], layer_sizes[3], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[3], num_classes))
        else:
            self.output = FastKAN([layer_sizes[3], num_classes], dropout=dropout_linear,
                                  first_dropout=True, grid_size=grid_size)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x


class EightSimpleFastConvKAN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            grid_size: int = 8,
            degree_out: int = 2,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(EightSimpleFastConvKAN, self).__init__()

        self.layers = nn.Sequential(
            FastKANConv2DLayer(input_channels, layer_sizes[0], grid_size=grid_size, kernel_size=3, groups=1, padding=1,
                               stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            L1(FastKANConv2DLayer(layer_sizes[0], layer_sizes[1], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), l1_penalty),
            L1(FastKANConv2DLayer(layer_sizes[1], layer_sizes[2], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), l1_penalty),
            L1(FastKANConv2DLayer(layer_sizes[2], layer_sizes[3], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), l1_penalty),
            L1(FastKANConv2DLayer(layer_sizes[3], layer_sizes[4], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), l1_penalty),
            L1(FastKANConv2DLayer(layer_sizes[4], layer_sizes[5], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), l1_penalty),
            L1(FastKANConv2DLayer(layer_sizes[5], layer_sizes[6], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), l1_penalty),
            L1(FastKANConv2DLayer(layer_sizes[6], layer_sizes[7], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = FastKAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                                  first_dropout=True, grid_size=grid_size)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x
