import torch
import torch.nn as nn

from kan_convs import KALNConv2DLayer

from utils import L1
import torch.nn.functional as F
from einops import einsum
from functools import lru_cache


class KALNLayer(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(self, input_features, output_features, degree=3, base_activation=nn.SiLU):
        super(KALNLayer, self).__init__()  # Initialize the parent nn.Module class

        self.input_features = input_features
        self.output_features = output_features
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # base_activation: Activation function used after each layer's computation
        self.base_activation = base_activation()

        # Base weight for linear transformation in each layer
        self.base_weight = nn.Parameter(torch.randn(output_features, input_features))
        # Polynomial weight for handling Legendre polynomial expansions
        self.poly_weight = nn.Parameter(torch.randn(output_features, input_features * (degree + 1)))
        # Layer normalization to stabilize learning and outputs
        self.layer_norm = nn.LayerNorm(output_features)

        # Initialize weights using Kaiming uniform distribution for better training start
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.poly_weight, nonlinearity='linear')

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Legendre polynomials
    def compute_legendre_polynomials(self, x, order):
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x  # P1 = x
        legendre_polys = [P0, P1]

        # Compute higher order polynomials using recurrence
        for n in range(1, order):
            Pn = ((2.0 * n + 1.0) * x * legendre_polys[-1] - n * legendre_polys[-2]) / (n + 1.0)
            legendre_polys.append(Pn)

        return torch.stack(legendre_polys, dim=-1)

    def forward(self, x):

        # Apply base activation to input and then linear transform with base weights
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        # Compute Legendre polynomials for the normalized x
        legendre_basis = self.compute_legendre_polynomials(x_normalized, self.polynomial_order)
        # Reshape legendre_basis to match the expected input dimensions for linear transformation
        legendre_basis = legendre_basis.view(x.size(0), -1)

        # Compute polynomial output using polynomial weights
        poly_output = F.linear(legendre_basis, self.poly_weight)
        # Combine base and polynomial outputs, normalize, and activate
        x = self.base_activation(self.layer_norm(base_output + poly_output))

        return x

class KALN(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0, degree=3,
                 base_activation=nn.SiLU, first_dropout: bool = True, **kwargs):
        super(KALN, self).__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            layer = KALNLayer(in_features, out_features, degree, base_activation=base_activation)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SimpleConvKALN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            degree: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(SimpleConvKALN, self).__init__()

        self.layers = nn.Sequential(
            KALNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1, padding=1, stride=1,
                            dilation=1, affine=affine, norm_layer=norm_layer),
            L1(KALNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KALNConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KALNConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[3], num_classes))
        else:
            self.output = KALN([layer_sizes[3], num_classes], dropout=dropout_linear, first_dropout=True,
                               degree=degree_out)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x


class EightSimpleConvKALN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            degree: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(EightSimpleConvKALN, self).__init__()

        self.layers = nn.Sequential(
            KALNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1, padding=1, stride=1,
                            dilation=1, affine=affine, norm_layer=norm_layer),
            L1(KALNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KALNConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KALNConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KALNConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KALNConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KALNConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            L1(KALNConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               l1_penalty),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KALN([layer_sizes[7], num_classes], dropout=dropout_linear, first_dropout=True,
                               degree=degree_out)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x
