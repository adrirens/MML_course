# Copyright 2024 Li, Ziyao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *


def linear_layer(input_dim, output_dim):
    linear = nn.Linear(input_dim, output_dim)
    nn.init.xavier_normal_(linear.weight)
    nn.init.constant_(linear.bias, 0)
    return linear

class MetaNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MetaNet, self).__init__()
        self.layers = nn.Sequential(
            linear_layer(input_dim, hidden_dim),
            nn.ReLU(),
            linear_layer(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

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
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
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
        use_layernorm: bool = True,
        base_activation: str = "silu",
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        self.num_grids = num_grids
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if base_activation == 'silu':
            base_activation = torch.nn.SiLU()
        elif base_activation == 'identity':
            base_activation = torch.nn.Identity()
        elif base_activation == 'zero':
            base_activation = lambda x: x*0.        
        if use_base_update:
            self.base_activation = base_activation


    def forward(self, x, weights, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:

            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        spline_weights = weights[:,:,:-1].reshape(self.output_dim, self.input_dim * self.num_grids)

        ret = F.linear(spline_basis.view(*spline_basis.shape[:-2], -1), spline_weights)
        if self.use_base_update:
            base_weights = weights[:,:,-1].reshape(self.output_dim, self.input_dim)
            base = F.linear(self.base_activation(x), base_weights)
            ret = ret + base
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2
    ):
        '''this function returns the learned curves in a FastKANLayer.
        input_index: the selected index of the input, in [0, input_dim) .
        output_index: the selected index of the output, in [0, output_dim) .
        num_pts: num of points sampled for the curve.
        num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve 
            will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
        '''
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]   # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )   # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y


class MetaFastKAN(nn.Module):
    def __init__(
            self, args
    ):
        super().__init__()
        self.embedding_dim = args.embedding_dim
        self.metanet = MetaNet(args.embedding_dim, args.hidden_dim, args.num_grids+1)
        self.embeddings = nn.ParameterList()        
        layers_hidden = [args.input_size] + args.layers_width + [args.output_size]
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=args.grid_min,
                grid_max=args.grid_max,
                num_grids=args.num_grids,
                use_base_update=args.use_base_update,
                base_activation=args.base_activation,
                spline_weight_init_scale=args.spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            embedding = nn.Parameter(torch.randn(in_features, out_features, args.embedding_dim))
            nn.init.xavier_normal_(embedding)
            self.embeddings.append(embedding)

    def forward(self, x):
        for l, layer in enumerate(self.layers):
            weights = self.metanet(self.embeddings[l]).reshape(layer.output_dim, layer.input_dim, -1)
            x = layer(x, weights)

        return x

