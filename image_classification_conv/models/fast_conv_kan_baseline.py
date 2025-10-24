import torch
import torch.nn as nn
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

class FastKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0, **norm_kwargs):
        super(FastKANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
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

        self.spline_conv = nn.ModuleList([conv_class(grid_size * input_dim // groups,
                                                     output_dim // groups,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     groups=1,
                                                     bias=False) for _ in range(groups)])

        self.layer_norm = nn.ModuleList([norm_class(input_dim // groups, **norm_kwargs) for _ in range(groups)])

        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size)

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        for conv_layer in self.spline_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_fast_kan(self, x, group_index):

        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](self.base_activation(x))
        if self.dropout is not None:
            x = self.dropout(x)
        spline_basis = self.rbf(self.layer_norm[group_index](x))
        spline_basis = spline_basis.moveaxis(-1, 2).flatten(1, 2)
        spline_output = self.spline_conv[group_index](spline_basis)
        x = base_output + spline_output

        return x

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_fast_kan(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class FastKANConv3DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0,
                 norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(FastKANConv3DLayer, self).__init__(nn.Conv3d, norm_layer,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=3,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout, **norm_kwargs)


class FastKANConv2DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0,
                 norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(FastKANConv2DLayer, self).__init__(nn.Conv2d, norm_layer,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=2,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout, **norm_kwargs)


class FastKANConv1DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0,
                 norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(FastKANConv1DLayer, self).__init__(nn.Conv1d, norm_layer,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=1,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout, **norm_kwargs)


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

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


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
    def __init__(self, args):
        super(SimpleFastConvKAN, self).__init__()
        layer_sizes = args.layer_sizes
        num_classes = args.num_classes
        input_channels = args.input_channel
        grid_size = args.grid_size
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
    def __init__(self, args):
        super(EightSimpleFastConvKAN, self).__init__()
        layer_sizes = args.layer_sizes
        num_classes = args.num_classes
        input_channels = args.input_channel
        grid_size = args.grid_size
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
