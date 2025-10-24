import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import abc



class WeightDecay(nn.Module):
    def __init__(self, module, weight_decay, name: str = None):
        if weight_decay < 0.0:
            raise ValueError(
                "Regularization's weight_decay should be greater than 0.0, got {}".format(
                    weight_decay
                )
            )

        super().__init__()
        self.module = module
        self.weight_decay = weight_decay
        self.name = name

        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    def remove(self):
        self.hook.remove()

    def _weight_decay_hook(self, *_):
        if self.name is None:
            for param in self.module.parameters():
                if param.grad is None or torch.all(param.grad == 0.0):
                    param.grad = self.regularize(param)
        else:
            for name, param in self.module.named_parameters():
                if self.name in name and (
                    param.grad is None or torch.all(param.grad == 0.0)
                ):
                    param.grad = self.regularize(param)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def extra_repr(self) -> str:
        representation = "weight_decay={}".format(self.weight_decay)
        if self.name is not None:
            representation += ", name={}".format(self.name)
        return representation

    @abc.abstractmethod
    def regularize(self, parameter):
        pass


class L2(WeightDecay):
    r"""Regularize module's parameters using L2 weight decay.

    Example::

        import torchlayers as tl

        # Regularize only weights of Linear module
        regularized_layer = tl.L2(tl.Linear(30), weight_decay=1e-5, name="weight")

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L2` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").

    """

    def regularize(self, parameter):
        return self.weight_decay * parameter.data


class L1(WeightDecay):
    """Regularize module's parameters using L1 weight decay.

    Example::

        import torchlayers as tl

        # Regularize all parameters of Linear module
        regularized_layer = tl.L1(tl.Linear(30), weight_decay=1e-5)

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L1` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").

    """

    def regularize(self, parameter):
        return self.weight_decay * torch.sign(parameter.data)

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


class KANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, spline_order, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0,
                 **norm_kwargs):
        super(KANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.grid_range = grid_range
        self.norm_kwargs = norm_kwargs

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        self.spline_conv = nn.ModuleList([conv_class((grid_size + spline_order) * input_dim // groups,
                                                     output_dim // groups,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     groups=1,
                                                     bias=False) for _ in range(groups)])

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(groups)])

        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        )
        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        for conv_layer in self.spline_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_kan(self, x, group_index):

        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](self.base_activation(x))

        x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations.
        # Compute the basis for the spline using intervals and input values.
        target = x.shape[1:] + self.grid.shape
        grid = self.grid.view(*list([1 for _ in range(self.ndim + 1)] + [-1, ])).expand(target).contiguous().to(
            x.device)

        bases = ((x_uns >= grid[..., :-1]) & (x_uns < grid[..., 1:])).to(x.dtype)

        # Compute the spline basis over multiple orders.
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[..., :-(k + 1)]
            right_intervals = grid[..., k:-1]
            delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
                                right_intervals - left_intervals)
            bases = ((x_uns - left_intervals) / delta * bases[..., :-1]) + \
                    ((grid[..., k + 1:] - x_uns) / (grid[..., k + 1:] - grid[..., 1:(-k)]) * bases[..., 1:])
        bases = bases.contiguous()
        bases = bases.moveaxis(-1, 2).flatten(1, 2)
        spline_output = self.spline_conv[group_index](bases)
        x = self.prelus[group_index](self.layer_norm[group_index](base_output + spline_output))

        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kan(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class KANConv3DLayer(KANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0, norm_layer=nn.InstanceNorm3d,
                 **norm_kwargs):
        super(KANConv3DLayer, self).__init__(nn.Conv3d, norm_layer,
                                             input_dim, output_dim,
                                             spline_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=3,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)


class KANConv2DLayer(KANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0, norm_layer=nn.InstanceNorm2d,
                 **norm_kwargs):
        super(KANConv2DLayer, self).__init__(nn.Conv2d, norm_layer,
                                             input_dim, output_dim,
                                             spline_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=2,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)


class KANConv1DLayer(KANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0, norm_layer=nn.InstanceNorm1d,
                 **norm_kwargs):
        super(KANConv1DLayer, self).__init__(nn.Conv1d, norm_layer,
                                             input_dim, output_dim,
                                             spline_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=1,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)


class SimpleConvKAN(nn.Module):
    def __init__(self, args):
        super(SimpleConvKAN, self).__init__()

        layer_sizes = args.layer_sizes
        num_classes = args.num_classes
        input_channels = args.input_channel
        spline_order = args.spline_order
        degree_out = args.degree_out
        groups = args.groups
        dropout = args.dropout
        dropout_linear = args.dropout_linear
        l1_penalty = args.l1_penalty
        affine = args.affine

        if args.norm_layer == 'batch':
            norm_layer = nn.BatchNorm2d
        elif args.norm_layer == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif args.norm_layer == 'layer':
            norm_layer = nn.LayerNorm
        else:
            NotImplementedError('Norm layer not implemented')


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
    def __init__(self, args):
        super(EightSimpleConvKAN, self).__init__()

        layer_sizes = args.layer_sizes
        num_classes = args.num_classes
        input_channels = args.input_channels
        spline_order = args.spline_order
        degree_out = args.degree_out
        groups = args.groups
        dropout = args.dropout
        dropout_linear = args.dropout_linear
        l1_penalty = args.l1_penalty
        affine = args.affine

        if args.norm_layer == 'batch':
            norm_layer = nn.BatchNorm2d
        elif args.norm_layer == 'instance':
            norm_layer = nn.InstanceNorm2d
        elif args.norm_layer == 'layer':
            norm_layer = nn.LayerNorm
        else:
            raise NotImplementedError('Norm layer not implemented')

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
