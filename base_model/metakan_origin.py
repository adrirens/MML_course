import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import math
from torchinfo import summary
from torch.profiler import profile, record_function, ProfilerActivity

def linear_layer(input_dim, output_dim):
    linear = nn.Linear(input_dim, output_dim)
    nn.init.xavier_normal_(linear.weight)
    nn.init.constant_(linear.bias, 0)
    return linear

class HyperNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HyperNetwork, self).__init__()
        self.layers = nn.Sequential(
            linear_layer(input_dim, hidden_dim),
            nn.ReLU(),
            linear_layer(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        base_activation=torch.nn.SiLU,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 样条阶数

        # 计算网格步长，并生成网格
        #         网格的作用
        # 定义B样条基函数的位置：

        # B样条基函数是在特定的支持点上进行计算的，这些支持点由网格确定。
        # 样条基函数在这些网格点上具有特定的值和形状。
        # 确定样条基函数的间隔：

        # 网格步长（h）决定了网格点之间的距离，从而影响样条基函数的平滑程度和覆盖范围。
        # 网格越密集，样条基函数的分辨率越高，可以更精细地拟合数据。
        # 构建用于插值和拟合的基础：

        # 样条基函数利用这些网格点进行插值，能够构建出连续的、平滑的函数。
        # 通过这些基函数，可以实现输入特征的复杂非线性变换。
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # 注册网格作为模型的buffer
        #         在PyTorch中，buffer是一种特殊类型的张量，它在模型中起到辅助作用，但不会作为模型参数进行更新。buffer通常用于存储一些在前向和后向传播过程中需要用到的常量或中间结果。buffer和模型参数一样，会被包含在模型的状态字典中（state dictionary），可以与模型一起保存和加载。

        # register_buffer 的作用
        # self.register_buffer("grid", grid) 的作用是将 grid 注册为模型的一个buffer。这样做有以下几个好处：

        # 持久化：buffer会被包含在模型的状态字典中，可以通过 state_dict 方法保存模型时一并保存，加载模型时也会一并恢复。这对于训练和推理阶段都很有用，确保所有相关的常量都能正确加载。

        # 无需梯度更新：buffer不会在反向传播过程中计算梯度和更新。它们是固定的，只在前向传播中使用。这对于像网格点这样的常量非常适合，因为这些点在训练过程中是固定的，不需要更新。

        # 易于使用：注册为 buffer 的张量可以像模型参数一样方便地访问和使用，而不必担心它们会被优化器错误地更新。

        # 初始化网络参数和超参数

        # 初始化基础权重参数，形状为 (out_features, in_features)



        # self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # 初始化样条权重参数，形状为 (out_features, in_features, grid_size + spline_order)
        # self.spline_weight = torch.nn.Parameter(
        #     torch.Tensor(out_features, in_features, grid_size + spline_order)
        # )

        # 如果启用了独立缩放样条功能，初始化样条缩放参数，形状为 (out_features, in_features)
        # if enable_standalone_scale_spline:
        #     self.spline_scaler = torch.nn.Parameter(
        #         torch.Tensor(out_features, in_features)
        #     )

        # # 噪声缩放系数，用于初始化样条权重时添加噪声
        # self.scale_noise = scale_noise

        # # 基础权重的缩放系数，用于初始化基础权重时的缩放因子
        # self.scale_base = scale_base

        # # 样条权重的缩放系数，用于初始化样条权重时的缩放因子
        # self.scale_spline = scale_spline

        # # 是否启用独立的样条缩放功能
        # self.enable_standalone_scale_spline = enable_standalone_scale_spline

        # 基础激活函数实例，用于对输入进行非线性变换
        self.base_activation = base_activation()




    def b_splines(self, x: torch.Tensor):
        """
        计算给定输入张量的B样条基函数。
        B样条（B-splines）是一种用于函数逼近和插值的基函数。
        它们具有局部性、平滑性和数值稳定性等优点，广泛应用于计算机图形学、数据拟合和机器学习中。
        在这段代码中，B样条基函数用于在输入张量上进行非线性变换，以提高模型的表达能力。
        在KAN（Kolmogorov-Arnold Networks）模型中，B样条基函数用于将输入特征映射到高维空间中，以便在该空间中进行线性变换。
        具体来说，B样条基函数能够在给定的网格点上对输入数据进行插值和逼近，从而实现复杂的非线性变换。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: B样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        # 确保输入张量的维度是2，并且其列数等于输入特征数
        assert x.dim() == 2 and x.size(1) == self.in_features

        # 获取网格点（包含在buffer中的self.grid）
        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)

        # 为了进行逐元素操作，将输入张量的最后一维扩展一维
        x = x.unsqueeze(-1)

        # 初始化B样条基函数的基矩阵
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        # 迭代计算样条基函数
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        # 确保B样条基函数的输出形状正确
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()



    def forward(self, x: torch.Tensor, weights):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        # 确保输入张量的最后一维大小等于输入特征数
        assert x.size(-1) == self.in_features
        weights = weights.view(self.out_features, self.in_features, -1)


        # 保存输入张量的原始形状
        original_shape = x.shape
        
        # 将输入张量展平为二维
        x = x.view(-1, self.in_features)

        # 计算基础线性变换的输出
        base_output = F.linear(self.base_activation(x), weights[:,:,-1])
        

        # 计算B样条基函数的输出
        spline_output = torch.einsum('bik,oik->bo', self.b_splines(x), weights[:,:,:self.grid_size + self.spline_order])
        
        # 合并基础输出和样条输出
        output = base_output + spline_output
        
        # 恢复输出张量的形状
        output = output.view(*original_shape[:-1], self.out_features)
        
        return output



########################################################## 类定义
class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        embedding_dim = 1,
        hidden_dim = 32,
        grid_size=5,
        spline_order=3,
        base_activation=torch.nn.SiLU,
        grid_range=[-1, 1],
    ):
        """
        初始化KAN模型。

        参数:
            layers_hidden (list): 每层的输入和输出特征数列表。
            grid_size (int): 网格大小。
            spline_order (int): 样条阶数。
            base_activation (nn.Module): 基础激活函数类。
            grid_range (list): 网格范围。
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 样条阶数
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.hidden_dim = hidden_dim
        self.embeddings = nn.ParameterList()
        self.hypernet = HyperNetwork(embedding_dim, hidden_dim, grid_size + spline_order+1)
        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            embedding = nn.Parameter(torch.randn(in_features, out_features,embedding_dim))
            nn,init.xavier_normal_(embedding)
            self.embeddings.append(embedding)  # 将嵌入参数添加到列表中
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation,
                    grid_range=grid_range,
                )
            )
    
    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否在前向传播过程中更新网格。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for l, layer in enumerate(self.layers):
            weights = self.hypernet(self.embeddings[l].view(-1, self.embedding_dim)) # 通过超网络生成权重
            x = layer(x, weights)
        return x