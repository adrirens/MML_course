import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init



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
        device = 'cpu'
    ):
        super(KANLinear, self).__init__()
        self.device = device    
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        self.base_activation = base_activation

        self.grid_range = torch.tensor(grid_range).unsqueeze(0).expand(in_features, -1)
        self.grid_range = self.grid_range.clone().to(dtype=torch.float32)
        self.grid_range = torch.nn.Parameter(self.grid_range).requires_grad_(False)

        self.grid_intervals = ((self.grid_range[:, 1] - self.grid_range[:, 0]) / grid_size)
        self.grid_intervals = torch.nn.Parameter(self.grid_intervals).requires_grad_(False)

        self.basis_matrix = self.calculate_basis_matrix()
        self.basis_matrix = torch.nn.Parameter(self.basis_matrix, requires_grad=False)

    def calculate_basis_matrix(self):
        """
        Compute the basis matrix for a uniform B-spline with a given spline degree.

        Returns:
            torch.Tensor: Basis matrix tensor.
        """

        basis_matrix = torch.tensor([
            [1]
        ], dtype=torch.float32)

        scalar = 1

        k = 2

        while k <= self.spline_order + 1:
            term_1 = torch.nn.functional.pad(basis_matrix, (0, 0, 0, 1), "constant", 0)
            term_3 = torch.nn.functional.pad(basis_matrix, (0, 0, 1, 0), "constant", 0)

            term_2 = torch.zeros((k - 1, k),  dtype=term_1.dtype)
            term_4 = torch.zeros((k - 1, k),  dtype=term_3.dtype)
            for i in range(k - 1):
                term_2[i, i] = i + 1
                term_2[i, i + 1] = k - (i + 2)

                term_4[i, i] = -1
                term_4[i, i + 1] = 1

            basis_matrix = torch.matmul(term_1, term_2) + torch.matmul(term_3, term_4)
            scalar *= 1 / (k - 1)
            k += 1

        basis_matrix *= scalar

        return basis_matrix.to(dtype=torch.float32)


    def power_bases(self, x: torch.Tensor):
        """
        Compute power bases for the given input tensor.

        Args:
            x (torch.Tensor):                   Input tensor.

        Returns:
            u (torch.Tensor):                   Power bases tensor.
            x_intervals (torch.Tensor):         Tensor representing the applicable grid interval for each input value.
        """

        # Determine applicable grid interval boundary values
        grid_floors = self.grid[:, 0]
        grid_floors = grid_floors.unsqueeze(0).expand(x.shape[0], -1)

        x = x.unsqueeze(dim=2)
        grid = self.grid.unsqueeze(dim=0)

        x_intervals = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
        x_interval_floor = torch.argmax(x_intervals.to(torch.int), dim=-1, keepdim=True)
        x_interval_floor = x_interval_floor.squeeze(-1)
        x_interval_floor = ((x_interval_floor * self.grid_intervals) + grid_floors)
        x_interval_ceiling = x_interval_floor + self.grid_intervals

        x = x.squeeze(2)

        # Calculate power bases
        u1_numerator = x - x_interval_floor
        u1_denominator = x_interval_ceiling - x_interval_floor
        u1 = (u1_numerator / u1_denominator).unsqueeze(-1)
        ones = torch.ones(u1.shape, dtype=x.dtype, device=self.device)
        u = torch.cat((ones, u1), -1)
        for i in range(2, self.spline_order + 1):
            base = u1 ** i
            u = torch.cat((u, base), -1)

        return u, x_intervals

    def b_splines_matrix(self, x):
        """
        Computes the b-spline output based on the given input tensor.

        Args:
            x (torch.Tensor):       Input tensor.

        Returns:
            result (torch.Tensor):   Tensor representing the outputs of each basis function.
        """

        # Calculate power bases and applicable grid intervals
        power_bases, x_intervals = self.power_bases(x)

        # Pad basis matrix per input
        basis_matrices = torch.nn.functional.pad(self.basis_matrix, (self.spline_order + self.grid_size, self.spline_order + self.grid_size),
                                                 mode='constant', value=0)
        basis_matrices = basis_matrices.unsqueeze(0).unsqueeze(0)
        basis_matrices = basis_matrices.expand(power_bases.size(0), self.in_features, -1, -1)

        # Calculate applicable grid intervals
        out_of_bounds_interval = torch.zeros((x_intervals.size(0), x_intervals.size(1), 1), dtype=torch.bool).to(device=self.device)
        x_intervals = torch.cat((out_of_bounds_interval, x_intervals), -1)

        # Identify and gather applicable basis functions
        basis_func_floor_indices = torch.argmax(x_intervals.to(torch.int), dim=-1, keepdim=True)
        basis_func_floor_indices = (2 * self.spline_order) + self.grid_size - basis_func_floor_indices + 1
        basis_func_indices = torch.arange(0, self.spline_order + self.grid_size, 1).unsqueeze(0).unsqueeze(0).to(device=self.device)
        basis_func_indices = basis_func_indices.expand(
            basis_matrices.size(0),
            basis_matrices.size(1),
            basis_matrices.size(2),
            -1
        )
        basis_func_indices = basis_func_indices.clone()
        basis_func_indices += basis_func_floor_indices.unsqueeze(-2).expand(-1, -1, basis_func_indices.size(-2), -1)

        basis_matrices = torch.gather(basis_matrices, -1, basis_func_indices)

        # Calculate basis function outputs
        power_bases = power_bases.unsqueeze(-2)
        result = torch.matmul(power_bases, basis_matrices)
        result = result.squeeze(-2)

        # in case grid is degenerate
        result = torch.nan_to_num(result)
        return result


    def b_splines_matrix_output(self, x: torch.Tensor):
        """
        Computes b-spline output based on the given input tensor and spline coefficients.

        Args:
            x (torch.Tensor):   Input tensor.

        Returns:
            result (torch.Tensor):   Tensor representing the outputs of each B-spline.
        """

        # Calculate basis function outputs
        basis_func_outputs = self.b_splines_matrix(x)


        result = torch.einsum('ijk,jlk->ijl', basis_func_outputs, self.scaled_spline_weight)

        return result

    def b_splines(self, x: torch.Tensor):

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        epsilon = 1e-8
        for k in range(1, self.spline_order + 1):
            delta_prev = grid[:, k:-1] - grid[:, : -(k + 1)]
            delta_next = grid[:, k + 1 :] - grid[:, 1:(-k)]
            term1 = (x - grid[:, : -(k + 1)]) / (delta_prev + epsilon) * bases[:, :, :-1]
            term2 = (grid[:, k + 1 :] - x) / (delta_next + epsilon) * bases[:, :, 1:]
            bases = term1 + term2

        return bases.contiguous()
    
    def forward(self, x: torch.Tensor, weight:torch.Tensor):
        assert x.size(-1) == self.in_features

        x_reshaped = x.view(-1, self.in_features) # (N, i)
        self.scaled_spline_weight = weight[:, :, :-1]
        spline_output = self.b_splines_matrix_output(x_reshaped)


        base_weight = weight[:,:,-1] # (N, i, o)
        base = self.base_activation(x_reshaped)
        base_output = base_weight[None, :, :] * base[:, :, None]

        y = base_output + spline_output

        y = torch.sum(y,dim=1)

        return y    



class MetaKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        embedding_dim=1,
        hidden_dim=16,
        grid_size=5,
        spline_order=3,
        base_activation=torch.nn.SiLU(), 
        grid_range=[-1, 1],
        device = 'cpu',
    ):      
        super(MetaKAN, self).__init__()      

        self.grid_size = grid_size
        self.spline_order = spline_order
        self.embedding_dim = embedding_dim        
        self.d_b = grid_size + spline_order + 1
        self.spline_dim = grid_size + spline_order


        self.embeddings = nn.ParameterList()

        self.hypernet = nn.ModuleList([
            nn.Sequential(
                linear_layer(embedding_dim, hidden_dim),
                nn.ReLU(), # Optional activation
                linear_layer(hidden_dim, self.d_b)
            ) for _ in range(len(layers_hidden) - 1)
        ])

        self.layers = nn.ModuleList()
        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            num_connections = in_features * out_features
            embedding = nn.Parameter(torch.randn(num_connections, embedding_dim))
            init.xavier_normal_(embedding)
            self.embeddings.append(embedding)

            kan_layer = KANLinear(
                 in_features,
                 out_features,
                 grid_size=grid_size,
                 spline_order=spline_order,
                 base_activation=base_activation, # Pass the module instance
                 grid_range=grid_range,
                 device=device
            )            
            self.layers.append(kan_layer)

    def forward(self, x: torch.Tensor):   

        for layer_index, (layer, embeddings_l, hypernet_l) in enumerate(zip(self.layers, self.embeddings, self.hypernet)):
            weight = hypernet_l(embeddings_l).view(layer.in_features, layer.out_features, self.d_b)
            x = layer(x, weight)

        return x


    def get_trainable_parameters(self):
        return list(self.embeddings.parameters()) + list(self.hypernet.parameters())


if __name__ == "__main__":
    # Example usage
    layers_hidden = [10, 20, 30]
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

    model = MetaKAN(layers_hidden,device=device)
    model.to(device)
    x = torch.randn(5, 10).to(device)  # Example input
    output = model(x)
    print(output.shape)  # Should match the output dimensions of the last layer
    print(output.device)