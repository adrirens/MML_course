import torch
import torch.nn as nn
from typing import List
import abc

import torch.optim as optim
import torchvision
import torch.nn.functional as F
from functools import lru_cache
from torch.nn.functional import conv3d, conv2d, conv1d
from einops import einsum



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


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y

class KACN(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0,
                 degree=3, first_dropout: bool = True, **kwargs):
        super(KACN, self).__init__()  # Initialize the parent nn.Module class

        # layers_hidden: A list of integers specifying the number of neurons in each layer
        self.layers_hidden = layers_hidden
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            # Base weight for linear transformation in each layer
            layer = ChebyKANLayer(in_features, out_features, degree)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MetaKACNConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, dropout=0.0, **norm_kwargs):
        super(MetaKACNConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs
        self.epsilon = 1e-7
        self.base_num = degree + 1
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


        arange_buffer_size = (1, 1, -1,) + tuple(1 for _ in range(ndim))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1).view(*arange_buffer_size))
        # Initialize weights using Kaiming uniform distribution for better training start



    def forward_kacn(self, x, group_index, layer_weight):

        # Apply base activation to input and then linear transform with base weights
        x = torch.tanh(x)
        x = torch.acos(torch.clamp(x, -1 + self.epsilon, 1 - self.epsilon)).unsqueeze(2)
        x = (x * self.arange).flatten(1, 2)
        x = x.cos()
        x = F.conv2d(x, layer_weight, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=1)
        x = self.layer_norm[group_index](x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def forward(self, x, layer_weight):

        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            layer_weight = layer_weight.view(self.groups, self.outdim // self.groups, (self.inputdim // self.groups) * (self.base_num), self.kernel_size, self.kernel_size)
            y = self.forward_kacn(_x, group_ind, layer_weight[group_ind])
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class MetaKACNConv3DLayer(MetaKACNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout=0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(MetaKACNConv3DLayer, self).__init__(nn.Conv3d, norm_layer,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=3, dropout=dropout, **norm_kwargs)


class MetaKACNConv2DLayer(MetaKACNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout=0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(MetaKACNConv2DLayer, self).__init__(nn.Conv2d, norm_layer,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=2, dropout=dropout, **norm_kwargs)


class MetaKACNConv1DLayer(MetaKACNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout=0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(MetaKACNConv1DLayer, self).__init__(nn.Conv1d, norm_layer,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=1, dropout=dropout, **norm_kwargs)




def linear_layer(input_dim, output_dim):
    linear = nn.Linear(input_dim, output_dim)
    nn.init.xavier_normal_(linear.weight)
    nn.init.constant_(linear.bias, 0)
    return linear

class HyperNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_hyper):
        super(HyperNetwork, self).__init__()
        self.layers = nn.Sequential(
            linear_layer(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_hyper),
            # linear_layer(hidden_dim, hidden_dim),
            # nn.ReLU(),
            linear_layer(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

class SimpleMetaConvKACN(nn.Module):
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
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            dropout_hyper: float = 0.0,
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(SimpleMetaConvKACN, self).__init__()
        self.groups = groups
        self.layer_sizes = [input_channels]+layer_sizes
        self.base_num = degree + 1
        self.hyper_net = HyperNetwork(input_dim=embedding_dim, output_dim=self.base_num, hidden_dim=hidden_dim,dropout_hyper=dropout_hyper)


        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1,input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(groups, layer_sizes[i] * layer_sizes[i + 1]*3*3//groups**2, embedding_dim)))        


        self.layers = nn.Sequential(
            MetaKACNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1, padding=1,
                            stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            MetaKACNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKACNConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKACNConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               

        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[3], num_classes))
        else:
            self.output = KACN([layer_sizes[3], num_classes], dropout=dropout_linear, first_dropout=True,
                               degree=degree_out)
        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            embedding = self.embeddings[i]
            layer_weight = self.hyper_net(embedding)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x

class EightSimpleMetaConvKACN(nn.Module):
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
            hidden_dim: int = 128,
            embedding_dim: int = 1,
            dropout_hyper: float = 0.0,
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(EightSimpleMetaConvKACN, self).__init__()

        self.groups = groups
        self.layer_sizes = [input_channels]+layer_sizes
        self.base_num = degree + 1
        self.hyper_net = HyperNetwork(input_dim=embedding_dim, output_dim=self.base_num, hidden_dim=hidden_dim,dropout_hyper=dropout_hyper)


        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1,input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(groups, layer_sizes[i] * layer_sizes[i + 1]*3*3//groups**2, embedding_dim)))        

        self.layers = nn.Sequential(
            MetaKACNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1, padding=1,
                            stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            MetaKACNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKACNConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKACNConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKACNConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKACNConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKACNConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKACNConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KACN([layer_sizes[7], num_classes], dropout=dropout_linear,
                               first_dropout=True, degree=degree_out)
        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            embedding = self.embeddings[i]
            layer_weight = self.hyper_net(embedding)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x
