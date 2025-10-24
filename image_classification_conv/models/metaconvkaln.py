import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv3d, conv2d, conv1d
from typing import List
import abc
from functools import lru_cache


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

class MetaKALNConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, conv_w_fun, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 ndim: int = 2, **norm_kwargs):
        super(MetaKALNConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = nn.SiLU()
        self.conv_w_fun = conv_w_fun
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs
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

        return torch.concatenate(legendre_polys, dim=1)

    def forward_kal(self, x, group_index, layer_weight):
        # Apply base activation to input and then linear transform with base weights

        base_weight = layer_weight[:, :self.inputdim // self.groups, :, :].view(self.outdim // self.groups, self.inputdim // self.groups, self.kernel_size, self.kernel_size)
        base_output = F.conv2d(self.base_activation(x), base_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)        

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1 if x.shape[0] > 0 else x

        if self.dropout is not None:
            x_normalized = self.dropout(x_normalized)

        # Compute Legendre polynomials for the normalized x
        legendre_basis = self.compute_legendre_polynomials(x_normalized, self.degree)
        # Reshape legendre_basis to match the expected input dimensions for linear transformation
        # Compute polynomial output using polynomial weights

        poly_weights = layer_weight[:, self.inputdim // self.groups:, :, :].reshape(self.outdim // self.groups, self.inputdim // self.groups * self.base_num, self.kernel_size, self.kernel_size) 
        
        poly_output = self.conv_w_fun(legendre_basis, poly_weights,
                                      stride=self.stride, dilation=self.dilation,
                                      padding=self.padding, groups=1)

        # poly_output = poly_output.view(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], self.outdim // self.groups)
        # Combine base and polynomial outputs, normalize, and activate
        x = base_output + poly_output
        if isinstance(self.layer_norm[group_index], nn.LayerNorm):
            orig_shape = x.shape
            x = self.layer_norm[group_index](x.view(orig_shape[0], -1)).view(orig_shape)
        else:
            x = self.layer_norm[group_index](x)
        x = self.base_activation(x)

        return x

    def forward(self, x, layer_weight):

        # x = self.base_conv(x)
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kal(_x, group_ind, layer_weight)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class MetaKALNConv3DLayer(MetaKALNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(MetaKALNConv3DLayer, self).__init__(nn.Conv3d, norm_layer, conv3d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=3, dropout=dropout, **norm_kwargs)


class MetaKALNConv2DLayer(MetaKALNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(MetaKALNConv2DLayer, self).__init__(nn.Conv2d, norm_layer, conv2d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=2, dropout=dropout, **norm_kwargs)


class MetaKALNConv1DLayer(MetaKALNConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(MetaKALNConv1DLayer, self).__init__(nn.Conv1d, norm_layer, conv1d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=1, dropout=dropout, **norm_kwargs)



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

class SimpleMetaConvKALN(nn.Module):
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
            dropout_hyper: float = 0.0,
            hidden_dim: int = 128,            
            norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super(SimpleMetaConvKALN, self).__init__()

        self.layer_sizes = [input_channels]+layer_sizes
        self.base_num = degree + 1
        self.hyper_net = HyperNetwork(input_dim=embedding_dim, output_dim=self.base_num+1, hidden_dim=hidden_dim,dropout_hyper=dropout_hyper)

        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))

            
        self.layers = nn.Sequential(
            MetaKALNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1, padding=1, stride=1,
                            dilation=1, affine=affine, norm_layer=norm_layer),
            MetaKALNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKALNConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKALNConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[3], num_classes))
        else:
            self.output = KALN([layer_sizes[3], num_classes], dropout=dropout_linear, first_dropout=True,
                               degree=degree_out)

        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            embedding = self.embeddings[i]
            layer_weight = self.hyper_net(embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.base_num+1), 3, 3)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x


class EightSimpleMetaConvKALN(nn.Module):
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
        super(EightSimpleMetaConvKALN, self).__init__()


        self.layer_sizes = [input_channels]+layer_sizes
        self.base_num = degree + 1
        self.hyper_net = HyperNetwork(input_dim=embedding_dim, output_dim=self.base_num+1, hidden_dim=hidden_dim,dropout_hyper=dropout_hyper)

        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(1,input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(groups, layer_sizes[i] * layer_sizes[i + 1]*3*3//groups**2, embedding_dim)))     

        self.layers = nn.Sequential(
            MetaKALNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1, padding=1, stride=1,
                            dilation=1, affine=affine, norm_layer=norm_layer),
            MetaKALNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKALNConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
               
            MetaKALNConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
              
            MetaKALNConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
            
            MetaKALNConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=2, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
              
            MetaKALNConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
              
            MetaKALNConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, degree=degree, groups=groups, padding=1,
                               stride=1, dilation=1, dropout=dropout, affine=affine, norm_layer=norm_layer),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = KALN([layer_sizes[7], num_classes], dropout=dropout_linear, first_dropout=True,
                               degree=degree_out)


        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            embedding = self.embeddings[i]
            layer_weight = self.hyper_net(embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.base_num+1), 3, 3)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x
