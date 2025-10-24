import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import abc
import numpy as np
from sklearn.cluster import KMeans


class MetaKANConvNDLayer_origin(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, spline_order, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0,
                 **norm_kwargs):
        super(MetaKANConvNDLayer_origin, self).__init__()
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
        self.grid_k = grid_size + spline_order
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


        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **norm_kwargs) for _ in range(groups)])

        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(groups)])

        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        )


    def forward_kan(self, x, group_index, layer_weight):

        # Apply base activation to input and then linear transform with base weights
        base_weight = layer_weight[:, -self.inputdim // self.groups:, :, :].reshape(self.outdim // self.groups, self.inputdim // self.groups, self.kernel_size, self.kernel_size)
        base_output = F.conv2d(self.base_activation(x), base_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

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
        spline_weight = layer_weight[:, :-self.inputdim // self.groups, :, :].reshape(self.outdim // self.groups, self.inputdim // self.groups * self.grid_k, self.kernel_size, self.kernel_size) 
        spline_output = F.conv2d(bases, spline_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

        x = self.prelus[group_index](self.layer_norm[group_index](base_output + spline_output))

        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def forward(self, x, layer_weight):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kan(_x, group_ind, layer_weight)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class MetaKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, spline_order, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0,
                 o_batch_size=None, 
                 **norm_kwargs):
        super(MetaKANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.spline_order = spline_order
        self.kernel_size = kernel_size # 保持原始类型 (int 或 tuple)
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation() if base_activation is not None else nn.Identity()
        self.grid_range = grid_range
        self.norm_kwargs = norm_kwargs # 原始保留
        self.grid_k = grid_size + spline_order
        
        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            elif ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            elif ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)
        
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        o_g = output_dim // groups


        if o_batch_size is None or o_batch_size <= 0:
            self.o_batch_size = o_g 
        else:
            self.o_batch_size = o_batch_size

        self.layer_norm = nn.ModuleList([norm_class(o_g, **norm_kwargs) for _ in range(groups)])
        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(groups)])

        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        )

    def forward_kan(self, x, group_index, layer_weight):
        o_g = self.outdim // self.groups
        i_g = self.inputdim // self.groups
        batch_size = x.shape[0]

        ks_tuple = []
        if isinstance(self.kernel_size, int):
            ks_tuple = (self.kernel_size,) * self.ndim
        elif isinstance(self.kernel_size, (list, tuple)) and len(self.kernel_size) == self.ndim:
            ks_tuple = tuple(self.kernel_size)
        else:

            if self.ndim == 2 and isinstance(self.kernel_size, int):
                 ks_tuple = (self.kernel_size, self.kernel_size)
            elif isinstance(self.kernel_size, tuple) and len(self.kernel_size) == self.ndim :
                 ks_tuple = self.kernel_size
            else:
                 raise ValueError(f"kernel_size {self.kernel_size} incompatible with ndim {self.ndim}")



        base_w_full = layer_weight[:, -i_g:, ...].reshape(o_g, i_g, *ks_tuple)
        spline_w_full = layer_weight[:, :-i_g, ...].reshape(o_g, i_g * self.grid_k, *ks_tuple)


        x_act = self.base_activation(x)


        x_uns = x.unsqueeze(-1)

        target_list = list(x.shape[1:])
        target_list.append(self.grid.shape[0])
        target = tuple(target_list)
        

        grid = self.grid.view(*list([1 for _ in range(self.ndim + 1)] + [-1, ])).expand(target).contiguous().to(x.device)

        bases = ((x_uns >= grid[..., :-1]) & (x_uns < grid[..., 1:])).to(x.dtype)
        for k_loop in range(1, self.spline_order + 1): 
            left_intervals = grid[..., :-(k_loop + 1)]
            right_intervals = grid[..., k_loop:-1]
            delta = right_intervals - left_intervals
            delta = torch.where(delta == 0, torch.ones_like(delta) * 1e-8, delta)
            
            g_k_plus_1 = grid[..., k_loop + 1:]
            g_1_minus_k = grid[..., 1:(-k_loop)]
            delta_next = g_k_plus_1 - g_1_minus_k
            delta_next = torch.where(delta_next == 0, torch.ones_like(delta_next) * 1e-8, delta_next)

            bases = ((x_uns - left_intervals) / delta * bases[..., :-1]) + \
                    ((g_k_plus_1 - x_uns) / delta_next * bases[..., 1:])
        bases = bases.contiguous()

        if self.ndim == 1: perm_dims = (0, 1, 3, 2)
        elif self.ndim == 2: perm_dims = (0, 1, 4, 2, 3)
        elif self.ndim == 3: perm_dims = (0, 1, 5, 2, 3, 4)
        else: raise ValueError("ndim must be 1, 2, or 3")
        bases = bases.permute(*perm_dims).contiguous().view(batch_size, i_g * self.grid_k, *x.shape[2:])


        out_sp_shape_list = []

        _s, _p, _d, _k = self.stride, self.padding, self.dilation, ks_tuple
        for d_idx in range(self.ndim):
            L_in = x.shape[2+d_idx]
            P_d = _p[d_idx] if isinstance(_p, (list,tuple)) else _p
            DIL_d = _d[d_idx] if isinstance(_d, (list,tuple)) else _d
            K_d = _k[d_idx] 
            S_d = _s[d_idx] if isinstance(_s, (list,tuple)) else _s
            L_out = (L_in + 2 * P_d - DIL_d * (K_d - 1) - 1) // S_d + 1
            out_sp_shape_list.append(L_out)
        out_sp_shape = tuple(out_sp_shape_list)
        # ---

        y_acc = torch.zeros( batch_size, o_g, *out_sp_shape, device=x.device, dtype=x.dtype )
        
        if self.ndim == 1: conv = F.conv1d
        elif self.ndim == 2: conv = F.conv2d
        elif self.ndim == 3: conv = F.conv3d
        else: raise ValueError(f"Unsupported ndim: {self.ndim}")

        for o_start in range(0, o_g, self.o_batch_size):
            o_end = min(o_start + self.o_batch_size, o_g)

            b_w_b = base_w_full[o_start:o_end, ...] 
            s_w_b = spline_w_full[o_start:o_end, ...] 

            b_out = conv(x_act, b_w_b, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
            s_out = conv(bases, s_w_b, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
            
            y_acc[:, o_start:o_end, ...] = b_out + s_out
        
        x = self.prelus[group_index](self.layer_norm[group_index](y_acc)) 

        if self.dropout is not None:
            x = self.dropout(x)

        return x

    def forward(self, x, layer_weight): 
        split_x = torch.split(x, self.inputdim // self.groups, dim=1) 
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kan(_x, group_ind, layer_weight)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y

    def calculate_spline_basis_maps(self, x_group: torch.Tensor) -> torch.Tensor:


        N_batch = x_group.shape[0]
        I_group = self.input_dim_per_group # x_group.shape[1]
        input_spatial_dims = x_group.shape[2:]

        grid_view_dims = [1] * (self.ndim + 1) + [-1] # e.g., [1,1,1,-1] for ndim=2
        # target_expand_shape: (I_group, *input_spatial_dims, G_total_points)
        target_expand_shape = list(x_group.shape[1:]) #获取 (I_group, *input_spatial_dims)
        target_expand_shape.append(self.grid.shape[0])


        grid_ready_for_broadcast = self.grid.view(*grid_view_dims).expand(target_expand_shape).contiguous().to(x_group.device)
        grid_ready_for_broadcast = grid_ready_for_broadcast.unsqueeze(0) # (1, I_group, *spatial_dims, G_total_points)


        x_uns = x_group.unsqueeze(-1) # (N, I_group, *input_spatial_dims, 1)
        
        bases = ((x_uns >= grid_ready_for_broadcast[..., :-1]) & (x_uns < grid_ready_for_broadcast[..., 1:])).to(x_group.dtype)

        epsilon = 1e-8 
        for k_order in range(1, self.spline_order + 1):
            g_left = grid_ready_for_broadcast[..., :-(k_order + 1)]
            g_right = grid_ready_for_broadcast[..., k_order:-1]
            
            delta_prev = g_right - g_left
            delta_prev = torch.where(delta_prev == 0, torch.ones_like(delta_prev) * epsilon, delta_prev) # Avoid division by zero

            g_k_plus_1 = grid_ready_for_broadcast[..., k_order + 1:]
            g_1_minus_k = grid_ready_for_broadcast[..., 1:(-k_order)]
            delta_next = g_k_plus_1 - g_1_minus_k
            delta_next = torch.where(delta_next == 0, torch.ones_like(delta_next) * epsilon, delta_next) # Avoid division by zero

            term1_num = x_uns - g_left
            term1 = (term1_num / delta_prev) * bases[..., :-1]
            
            term2_num = g_k_plus_1 - x_uns
            term2 = (term2_num / delta_next) * bases[..., 1:]
            bases = term1 + term2
        
        bases = bases.contiguous() # Shape: (N, I_group, *input_spatial_dims, self.grid_k)
        
        permute_dims = [0, 1, self.ndim + 2] + list(range(2, self.ndim + 2))
        bases_permuted = bases.permute(*permute_dims).contiguous()
        
        final_shape_channels = I_group * self.grid_k
        bases_reshaped = bases_permuted.view(N_batch, final_shape_channels, *input_spatial_dims)
        
        return bases_reshaped

class MetaKANConv3DLayer(MetaKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0, norm_layer=nn.InstanceNorm3d,
                 **norm_kwargs):
        super(MetaKANConv3DLayer, self).__init__(nn.Conv3d, norm_layer,
                                             input_dim, output_dim,
                                             spline_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=3,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)


class MetaKANConv2DLayer(MetaKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0, norm_layer=nn.InstanceNorm2d,
                 **norm_kwargs):
        super(MetaKANConv2DLayer, self).__init__(nn.Conv2d, norm_layer,
                                             input_dim, output_dim,
                                             spline_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=2,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)


class MetaKANConv1DLayer(MetaKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0, norm_layer=nn.InstanceNorm1d,
                 **norm_kwargs):
        super(MetaKANConv1DLayer, self).__init__(nn.Conv1d, norm_layer,
                                             input_dim, output_dim,
                                             spline_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=1,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)


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

def linear_layer(input_dim, output_dim):
    linear = nn.Linear(input_dim, output_dim)
    nn.init.xavier_normal_(linear.weight)
    nn.init.constant_(linear.bias, 0)
    return linear

class HyperNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,dropout_hyper=0.0):
        super(HyperNetwork, self).__init__()
        self.layers = nn.Sequential(
            linear_layer(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_hyper),
            linear_layer(hidden_dim, output_dim)
            # linear_layer(hidden_dim, hidden_dim//2),
            # nn.ReLU(),
            # linear_layer(hidden_dim//2, hidden_dim//4),
            # nn.ReLU(),
            # linear_layer(hidden_dim//4, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

class SimpleMetaConvKAN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            grid_size: int = 5,
            spline_order: int = 3,
            degree_out: int = 2,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            embedding_dim: int = 1,
            hidden_dim: int = 128,            
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(SimpleMetaConvKAN, self).__init__()
        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size + spline_order

        self.hyper_net = HyperNetwork(input_dim=embedding_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim)

        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))
        self.layers = nn.Sequential(
            MetaKANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, spline_order=spline_order, groups=1,
                           padding=1, stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            MetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            

        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))        
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[3], num_classes))
        else:

            self.output = KAN([layer_sizes[3], num_classes], dropout=dropout_linear,
                              first_dropout=True, spline_order=spline_order)
        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            embedding = self.embeddings[i]
            layer_weight = self.hyper_net(embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.grid_k+1), 3, 3)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x

class EightSimpleMetaConvKAN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            grid_size: int = 5,
            spline_order: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            dropout_hyper: float = 0.0,           
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(EightSimpleMetaConvKAN, self).__init__()
        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size + spline_order

        self.hyper_net = HyperNetwork(input_dim=embedding_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim, dropout_hyper=dropout_hyper)

        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))
        self.layers = nn.Sequential(
            MetaKANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, spline_order=spline_order, groups=1,
                           padding=1, stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            MetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               

        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                              first_dropout=True, spline_order=spline_order)
        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            embedding = self.embeddings[i]
            layer_weight = self.hyper_net(embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.grid_k+1), 3, 3)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x
    
class EightSimpleMetaConvKAN_M(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            n_metanets: int = 1,
            grid_size: int = 5,
            spline_order: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            dropout_hyper: float = 0.0,           
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(EightSimpleMetaConvKAN_M, self).__init__()

        feats  = np.array(layer_sizes).reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_metanets, random_state=42).fit(feats)
        labels = kmeans.labels_        

        centers = np.argsort(kmeans.cluster_centers_.squeeze())
        reorder = {old:new for new, old in enumerate(centers)}
        self.layer2hyper = [reorder[int(lab)] for lab in labels]  # list 长度 = L        
        print(self.layer2hyper)


        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size + spline_order

        self.hyper_net = nn.ModuleList([
            HyperNetwork(input_dim=embedding_dim,
                         output_dim=self.grid_k + 1,
                         hidden_dim=hidden_dim,
                         dropout_hyper=dropout_hyper)
            for _ in range(n_metanets)
        ])


        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))
        self.layers = nn.Sequential(
            MetaKANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, spline_order=spline_order, groups=1,
                           padding=1, stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            MetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               

        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                              first_dropout=True, spline_order=spline_order)
        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)


    def forward(self, x):
        for i, (layer, emb) in enumerate(zip(self.layers, self.embeddings)):
            hnet_id   = self.layer2hyper[i]
            weight    = self.hyper_net[hnet_id](emb).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.grid_k+1), 3, 3)
            x = layer(x, weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x


class EightSimpleMetaConvKAN_L(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            grid_size: int = 5,
            spline_order: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            dropout_hyper: float = 0.0,           
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(EightSimpleMetaConvKAN_L, self).__init__()
        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size + spline_order

        self.hyper_net = HyperNetwork(input_dim=embedding_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim, dropout_hyper=dropout_hyper)

        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))
        self.layers = nn.Sequential(
            MetaKANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, spline_order=spline_order, groups=1,
                           padding=1, stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            MetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               

        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                              first_dropout=True, spline_order=spline_order)
        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            embedding = self.embeddings[i]
            layer_weight = self.hyper_net(embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.grid_k+1), 3, 3)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x
   

class EightSimpleMetaConvKAN_L3(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            grid_size: int = 5,
            spline_order: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            dropout_hyper: float = 0.0,           
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(EightSimpleMetaConvKAN_L3, self).__init__()
        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size + spline_order

        self.hyper_net = nn.ModuleList([HyperNetwork(input_dim=embedding_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim, dropout_hyper = dropout_hyper) for _ in range(3)])
        


        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))
        self.layers = nn.Sequential(
            MetaKANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, spline_order=spline_order, groups=1,
                           padding=1, stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            MetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               

        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                              first_dropout=True, spline_order=spline_order)
        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            embedding = self.embeddings[i]
            if i <4 :
                hypernet = self.hyper_net[0]
            elif i < 6:
                hypernet = self.hyper_net[1]
            elif i < 8:
                hypernet = self.hyper_net[2]

            
            layer_weight = hypernet(embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.grid_k+1), 3, 3)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x        


class EightSimpleMetaConvKAN_L4(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            grid_size: int = 5,
            spline_order: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            dropout_hyper: float = 0.0,           
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(EightSimpleMetaConvKAN_L4, self).__init__()
        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size + spline_order

        self.hyper_net = nn.ModuleList([HyperNetwork(input_dim=embedding_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim, dropout_hyper = dropout_hyper) for _ in range(4)])
        


        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))
        self.layers = nn.Sequential(
            MetaKANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, spline_order=spline_order, groups=1,
                           padding=1, stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            MetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               

        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                              first_dropout=True, spline_order=spline_order)
        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_normal_(embedding)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            embedding = self.embeddings[i]
            if i <2 :
                hypernet = self.hyper_net[0]
            elif i < 4:
                hypernet = self.hyper_net[1]
            elif i < 6:
                hypernet = self.hyper_net[2]
            else:
                hypernet = self.hyper_net[3]

            
            layer_weight = hypernet(embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.grid_k+1), 3, 3)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x                
    

class EightSimpleMetaConvKAN_L5(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            grid_size: int = 5,
            spline_order: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            dropout_hyper: float = 0.0,           
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(EightSimpleMetaConvKAN_L5, self).__init__()
        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size + spline_order

        self.hyper_net = nn.ModuleList([HyperNetwork(input_dim=embedding_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim, dropout_hyper = dropout_hyper) for _ in range(5)])
        


        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))
        self.layers = nn.Sequential(
            MetaKANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, spline_order=spline_order, groups=1,
                           padding=1, stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            MetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               

        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                              first_dropout=True, spline_order=spline_order)
        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            embedding = self.embeddings[i]
            if i <2 :
                hypernet = self.hyper_net[0]
            elif i < 4:
                hypernet = self.hyper_net[1]
            elif i < 5:
                hypernet = self.hyper_net[2]
            elif i<6:
                hypernet = self.hyper_net[3]
            else:
                hypernet = self.hyper_net[4]

            
            layer_weight = hypernet(embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.grid_k+1), 3, 3)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x        
    

class EightSimpleMetaConvKAN_L6(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            grid_size: int = 5,
            spline_order: int = 3,
            degree_out: int = 3,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            dropout_hyper: float = 0.0,           
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(EightSimpleMetaConvKAN_L6, self).__init__()
        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size + spline_order

        self.hyper_net = nn.ModuleList([HyperNetwork(input_dim=embedding_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim, dropout_hyper = dropout_hyper) for _ in range(6)])
        


        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))
        self.layers = nn.Sequential(
            MetaKANConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, spline_order=spline_order, groups=1,
                           padding=1, stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            MetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKANConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKANConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, spline_order=spline_order, groups=groups,
                              padding=1, stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               

        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                              first_dropout=True, spline_order=spline_order)
        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            embedding = self.embeddings[i]
            if i <2 :
                hypernet = self.hyper_net[0]
            elif i < 4:
                hypernet = self.hyper_net[1]
            else:
                hypernet = self.hyper_net[i-3]

            
            layer_weight = hypernet(embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.grid_k+1), 3, 3)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x                
        