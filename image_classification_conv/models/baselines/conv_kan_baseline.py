import torch
import torch.nn as nn

from kan_convs import KANConv2DLayer

from utils import L1
import torch.nn.functional as F
from typing import List, Optional

class KAN(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(self, layers_hidden, dropout: float = 0.0, grid_size=5, spline_order=3, base_activation=nn.GELU,
                 grid_range: List = [-1, 1], l1_decay: float = 0.0, first_dropout: bool = True, **kwargs):
        super(KAN, self).__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation
        self.grid_range = grid_range

        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = KANLayer(in_features, out_features, grid_size=grid_size, spline_order=spline_order,
                             base_activation=base_activation, grid_range=grid_range)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class KANLayer(nn.Module):
    def __init__(self, input_features, output_features, grid_size=5, spline_order=3, base_activation=nn.GELU,
                 grid_range=[-1, 1]):
        super(KANLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # The number of points in the grid for the spline interpolation.
        self.grid_size = grid_size
        # The order of the spline used in the interpolation.
        self.spline_order = spline_order
        # Activation function used for the initial transformation of the input.
        self.base_activation = base_activation()
        # The range of values over which the grid for spline interpolation is defined.
        self.grid_range = grid_range

        # Initialize the base weights with random values for the linear transformation.
        self.base_weight = nn.Parameter(torch.randn(output_features, input_features))
        # Initialize the spline weights with random values for the spline transformation.
        self.spline_weight = nn.Parameter(torch.randn(output_features, input_features, grid_size + spline_order))
        # Add a layer normalization for stabilizing the output of this layer.
        self.layer_norm = nn.LayerNorm(output_features)
        # Add a PReLU activation for this layer to provide a learnable non-linearity.
        self.prelu = nn.PReLU()

        # Compute the grid values based on the specified range and grid size.
        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        ).expand(input_features, -1).contiguous()

        # Initialize the weights using Kaiming uniform distribution for better initial values.
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.spline_weight, nonlinearity='linear')

    def forward(self, x):
        # Process each layer using the defined base weights, spline weights, norms, and activations.
        grid = self.grid.to(x.device)
        # Move the input tensor to the device where the weights are located.

        # Perform the base linear transformation followed by the activation function.
        base_output = F.linear(self.base_activation(x), self.base_weight)
        x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations.
        # Compute the basis for the spline using intervals and input values.
        bases = ((x_uns >= grid[:, :-1]) & (x_uns < grid[:, 1:])).to(x.dtype).to(x.device)

        # Compute the spline basis over multiple orders.
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[:, :-(k + 1)]
            right_intervals = grid[:, k:-1]
            delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
                                right_intervals - left_intervals)
            bases = ((x_uns - left_intervals) / delta * bases[:, :, :-1]) + \
                    ((grid[:, k + 1:] - x_uns) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        bases = bases.contiguous()

        # Compute the spline transformation and combine it with the base transformation.
        spline_output = F.linear(bases.view(x.size(0), -1), self.spline_weight.view(self.spline_weight.size(0), -1))
        # Apply layer normalization and PReLU activation to the combined output.
        x = self.prelu(self.layer_norm(base_output + spline_output))

        return x

class SimpleConvKAN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            spline_order: int = 3,
            degree_out: int = 2,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.InstanceNorm2d
    ):
        super(SimpleConvKAN, self).__init__()

        self.layers = nn.Sequential(
            KANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, spline_order=spline_order, groups=1,
                           padding=1, stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            L1(KANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[3], num_classes))
        else:

            self.output = KAN([layer_sizes[3], num_classes], dropout=dropout_linear,
                              first_dropout=True, spline_order=spline_order)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x


class EightSimpleConvKAN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            spline_order: int = 5,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(EightSimpleConvKAN, self).__init__()

        self.layers = nn.Sequential(
            KANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, spline_order=spline_order, groups=1,
                           padding=1, stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            L1(KANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KANConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                              first_dropout=True, spline_order=spline_order)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x
