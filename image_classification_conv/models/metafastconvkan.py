import torch
import torch.nn as nn
from typing import List
import abc

import torch.optim as optim
import torchvision
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


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




class L1(WeightDecay):
    """Regularize module's parameters using L1 weight decay.

    Example::

        import torchlayers as tl
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

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

class FastMetaKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0, **norm_kwargs):
        super(FastMetaKANConvNDLayer, self).__init__()
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


    def forward_fast_kan(self, x, group_index,layer_weight):

        # Apply base activation to input and then linear transform with base weights
        base_weight = layer_weight[:, -self.inputdim // self.groups:, :, :].view(self.outdim // self.groups, self.inputdim // self.groups, self.kernel_size, self.kernel_size)
        base_output = F.conv2d(self.base_activation(x), base_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
        if self.dropout is not None:
            x = self.dropout(x)
        spline_weight = layer_weight[:, :-self.inputdim // self.groups, :, :].reshape(self.outdim // self.groups, self.inputdim // self.groups * self.grid_size, self.kernel_size, self.kernel_size)    
        spline_basis = self.rbf(self.layer_norm[group_index](x))
        spline_basis = spline_basis.moveaxis(-1, 2).flatten(1, 2)
        spline_output = F.conv2d(spline_basis, spline_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

        x = base_output + spline_output

        return x

    def forward(self, x, layer_weight):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_fast_kan(_x, group_ind, layer_weight)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y
    


class FastMetaKANConv2DLayer(FastMetaKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-1, 1], dropout=0.0,
                 norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(FastMetaKANConv2DLayer, self).__init__(nn.Conv2d, norm_layer,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=2,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout, **norm_kwargs)
        


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


class SimpleMetaFastConvKAN(nn.Module):
    def __init__(
            self,
            layer_sizes:list = [8 * 4, 16 * 4, 32 * 4, 64 * 4],
            num_classes: int = 10,
            input_channels: int = 3,
            grid_size: int = 8,
            degree_out: int = 2,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            drop_hyper: float = 0.0,
            norm_layer: nn.Module = nn.InstanceNorm2d
    ):
        super(SimpleMetaFastConvKAN, self).__init__()
        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size

        self.hyper_net = HyperNetwork(input_dim=embedding_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim,dropout_hyper=drop_hyper)

        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))

        self.layers = nn.ModuleList([
            FastMetaKANConv2DLayer(input_channels, layer_sizes[0], grid_size=grid_size, kernel_size=3, groups=1, padding=1,
                               stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), 
            FastMetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), 
            FastMetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), 
        ])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[3], num_classes))
        else:
            self.output = FastKAN([layer_sizes[3], num_classes], dropout=dropout_linear,
                                  first_dropout=True, grid_size=grid_size)

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







class EightFastMetaConvKAN(nn.Module):
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
            norm_layer: nn.Module = nn.BatchNorm2d,
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            dropout_hyper: float = 0.0
    ):
        super(EightFastMetaConvKAN, self).__init__()
        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size       
        self.hyper_net = HyperNetwork(input_dim=embedding_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim,dropout_hyper = dropout_hyper)

        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))
        self.layers = nn.Sequential(
            FastMetaKANConv2DLayer(input_channels, layer_sizes[0], grid_size=grid_size, kernel_size=3, groups=1, padding=1,
                               stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[3], layer_sizes[4], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[4], layer_sizes[5], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[5], layer_sizes[6], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), 
            FastMetaKANConv2DLayer(layer_sizes[6], layer_sizes[7], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = FastKAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                                  first_dropout=True, grid_size=grid_size)
            
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
    

class EightFastMetaConvKAN_M(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            n_metanets: int = 1,
            input_channels: int = 1,
            grid_size: int = 8,
            degree_out: int = 2,
            groups: int = 1,
            dropout: float = 0.0,
            dropout_linear: float = 0.0,
            l1_penalty: float = 0.0,
            affine: bool = True,
            norm_layer: nn.Module = nn.BatchNorm2d,
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            dropout_hyper: float = 0.0
    ):
        super(EightFastMetaConvKAN_M, self).__init__()

        feats  = np.array(layer_sizes).reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_metanets, random_state=42).fit(feats)
        labels = kmeans.labels_        

        centers = np.argsort(kmeans.cluster_centers_.squeeze())
        reorder = {old:new for new, old in enumerate(centers)}
        self.layer2hyper = [reorder[int(lab)] for lab in labels]  # list 长度 = L        
        print(self.layer2hyper)


        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size       
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
            FastMetaKANConv2DLayer(input_channels, layer_sizes[0], grid_size=grid_size, kernel_size=3, groups=1, padding=1,
                               stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[3], layer_sizes[4], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[4], layer_sizes[5], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[5], layer_sizes[6], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), 
            FastMetaKANConv2DLayer(layer_sizes[6], layer_sizes[7], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = FastKAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                                  first_dropout=True, grid_size=grid_size)
            
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

class EightFastMetaConvKAN_DE(nn.Module):
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
            norm_layer: nn.Module = nn.BatchNorm2d,
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            dropout_hyper: float = 0.0,
            layer_emb_dim: int = 1
    ):
        super(EightFastMetaConvKAN_DE, self).__init__()
        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size       
        self.hyper_net = HyperNetwork(input_dim=embedding_dim+layer_emb_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim,dropout_hyper = dropout_hyper)

        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))

        self.layer_embeddings = nn.ModuleList([nn.Embedding(1, layer_emb_dim) for i in range(len(layer_sizes))])


        self.layers = nn.Sequential(
            FastMetaKANConv2DLayer(input_channels, layer_sizes[0], grid_size=grid_size, kernel_size=3, groups=1, padding=1,
                               stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[3], layer_sizes[4], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[4], layer_sizes[5], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[5], layer_sizes[6], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), 
            FastMetaKANConv2DLayer(layer_sizes[6], layer_sizes[7], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = FastKAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                                  first_dropout=True, grid_size=grid_size)
            
        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)
        for embedding in self.layer_embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            embedding = self.embeddings[i]
            layer_embedding = self.layer_embeddings[i](torch.tensor([0]).to(x.device)).expand(embedding.size(0), -1)
            embedding = torch.cat((embedding, 0.01 * layer_embedding), dim=-1)       
            # embedding = embedding + layer_embedding     
            layer_weight = self.hyper_net(embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.grid_k+1), 3, 3)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x



class EightFastMetaConvKAN_L3(nn.Module):
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
            norm_layer: nn.Module = nn.BatchNorm2d,
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            dropout_hyper: float = 0.0
    ):
        super(EightFastMetaConvKAN_L3, self).__init__()
        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size       
        # self.hyper_net = HyperNetwork(input_dim=embedding_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim,dropout_hyper = dropout_hyper)
        self.hyper_net = nn.ModuleList(
            [HyperNetwork(input_dim=embedding_dim,output_dim=self.grid_k+1, hidden_dim=hidden_dim, dropout_hyper=dropout_hyper) for _ in range(3)]
        )
        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))
        self.layers = nn.Sequential(
            FastMetaKANConv2DLayer(input_channels, layer_sizes[0], grid_size=grid_size, kernel_size=3, groups=1, padding=1,
                               stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[3], layer_sizes[4], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[4], layer_sizes[5], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[5], layer_sizes[6], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), 
            FastMetaKANConv2DLayer(layer_sizes[6], layer_sizes[7], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = FastKAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                                  first_dropout=True, grid_size=grid_size)
            
        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)


    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i<4:
                hypernet = self.hyper_net[0]
            elif i<6:
                hypernet = self.hyper_net[1]
            else:
                hypernet = self.hyper_net[2]

            embedding = self.embeddings[i]
            layer_weight = hypernet(embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.grid_k+1), 3, 3)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x


class EightFastMetaConvKAN_L3_DE(nn.Module):
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
            norm_layer: nn.Module = nn.BatchNorm2d,
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            dropout_hyper: float = 0.0,
            layer_emb_dim: int = 1
    ):
        super(EightFastMetaConvKAN_L3_DE, self).__init__()
        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size       
        # self.hyper_net = HyperNetwork(input_dim=embedding_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim,dropout_hyper = dropout_hyper)
        self.hyper_net = nn.ModuleList(
            [HyperNetwork(input_dim=embedding_dim+layer_emb_dim,output_dim=self.grid_k+1, hidden_dim=hidden_dim, dropout_hyper=dropout_hyper) for _ in range(3)]
        )


        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))

        self.layer_embeddings = nn.ModuleList([nn.Embedding(1, layer_emb_dim) for i in range(len(layer_sizes))])

        self.layers = nn.Sequential(
            FastMetaKANConv2DLayer(input_channels, layer_sizes[0], grid_size=grid_size, kernel_size=3, groups=1, padding=1,
                               stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[3], layer_sizes[4], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[4], layer_sizes[5], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[5], layer_sizes[6], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), 
            FastMetaKANConv2DLayer(layer_sizes[6], layer_sizes[7], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = FastKAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                                  first_dropout=True, grid_size=grid_size)
            
        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)
        for embedding in self.layer_embeddings:
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i<4:
                hypernet = self.hyper_net[0]
            elif i<6:
                hypernet = self.hyper_net[1]
            else:
                hypernet = self.hyper_net[2]

            embedding = self.embeddings[i]
            layer_embedding = self.layer_embeddings[i](torch.tensor([0]).to(x.device)).expand(embedding.size(0), -1)
            embedding = torch.cat((embedding, 0.01*layer_embedding), dim=-1)
            layer_weight = hypernet(embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.grid_k+1), 3, 3)            

            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x

class EightFastMetaConvKAN_L(nn.Module):
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
            norm_layer: nn.Module = nn.BatchNorm2d,
            embedding_dim: int = 1,
            hidden_dim: int = 128,
            dropout_hyper: float = 0.0
    ):
        super(EightFastMetaConvKAN_L, self).__init__()
        self.layer_sizes = [input_channels]+layer_sizes
        self.grid_k = grid_size       
        # self.hyper_net = HyperNetwork(input_dim=embedding_dim, output_dim=self.grid_k+1, hidden_dim=hidden_dim,dropout_hyper = dropout_hyper)
        self.hyper_net = nn.ModuleList(
            [HyperNetwork(input_dim=embedding_dim,output_dim=self.grid_k+1, hidden_dim=hidden_dim,dropout_hyper= dropout_hyper) for _ in range(len(layer_sizes))]
        )
        # 初始化每一层的嵌入向量列表 nn.ParameterList
        self.embeddings = nn.ParameterList([nn.Parameter(torch.randn(input_channels * layer_sizes[0]*3*3, embedding_dim))])
        for i in range(len(layer_sizes) - 1):
            self.embeddings.append(nn.Parameter(torch.randn(layer_sizes[i] * layer_sizes[i + 1]*3*3, embedding_dim)))
        self.layers = nn.Sequential(
            FastMetaKANConv2DLayer(input_channels, layer_sizes[0], grid_size=grid_size, kernel_size=3, groups=1, padding=1,
                               stride=1, dilation=1, affine=affine, norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[0], layer_sizes[1], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[1], layer_sizes[2], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[2], layer_sizes[3], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[3], layer_sizes[4], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[4], layer_sizes[5], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=2, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer),
            FastMetaKANConv2DLayer(layer_sizes[5], layer_sizes[6], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer), 
            FastMetaKANConv2DLayer(layer_sizes[6], layer_sizes[7], grid_size=grid_size, kernel_size=3, groups=groups,
                                  padding=1, stride=1, dilation=1, dropout=dropout, affine=affine,
                                  norm_layer=norm_layer)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if degree_out < 2:
            self.output = nn.Sequential(nn.Dropout(p=dropout_linear), nn.Linear(layer_sizes[7], num_classes))
        else:
            self.output = FastKAN([layer_sizes[7], num_classes], dropout=dropout_linear,
                                  first_dropout=True, grid_size=grid_size)
            
        self._initialize_embeddings()


    def _initialize_embeddings(self):
        """ 对嵌入向量进行 Xavier 初始化 """
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding)


    def forward(self, x):
        for i, layer in enumerate(self.layers):


            embedding = self.embeddings[i]
            layer_weight = self.hyper_net[i](embedding).reshape(self.layer_sizes[i+1], self.layer_sizes[i]*(self.grid_k+1), 3, 3)
            x = layer(x, layer_weight)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x
