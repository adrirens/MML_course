import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init



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



class MetaKAN(torch.nn.Module):
    def __init__(
        self,
        args,
    ):      
        super(MetaKAN, self).__init__()      
        layers_hidden = [args.input_size] + args.layers_width + [args.output_size]
        self.grid_size = args.grid_size
        self.spline_order = args.spline_order
        self.embedding_dim = args.embedding_dim        
        self.d_b = args.grid_size + args.spline_order + 1
        self.spline_dim = args.grid_size + args.spline_order
        self.o_batch_size = args.o_batch_size


        self.embeddings = nn.ParameterList()

        self.metanet = nn.ModuleList([
            MetaNet(args.input_size, args.hidden_dim, self.d_b) for _ in range(len(layers_hidden) - 1)
        ])

        self.layers = nn.ModuleList()
        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            num_connections = in_features * out_features
            embedding = nn.Parameter(torch.randn(num_connections, args.embedding_dim))
            init.xavier_normal_(embedding)
            self.embeddings.append(embedding)

            kan_layer = KANLinear(
                 in_features,
                 out_features,
                 grid_size=args.grid_size,
                 spline_order=args.spline_order,
                 base_activation=args.base_activation, # Pass the module instance
                 grid_range=args.grid_range,
            )            
            self.layers.append(kan_layer)

    def forward(self, x: torch.Tensor):
        original_shape_prefix = x.shape[:-1]
        current_x = x        

        for layer, embeddings_l, metanet_l in zip(self.layers, self.embeddings, self.metanet):
            in_features = layer.in_features
            out_features = layer.out_features
            N_conn = in_features * out_features


            x = current_x.reshape(-1, in_features) # Shape: (N, input)
            N = x.shape[0]      

            spline_basis = layer.b_splines(x) # Shape: (N, input, G+k)      
            spline_basis = spline_basis.reshape(N, -1)

            base_feature = layer.base_activation(x) # Shape: (N, i)

            layer_output = torch.zeros(N, out_features, device=current_x.device, dtype=current_x.dtype)

            for o_start in range(0, out_features, self.o_batch_size):
                o_end = min(o_start + self.o_batch_size, out_features)
                current_batch_o_size = o_end - o_start # b                

                emb_start_idx = o_start * in_features
                emb_end_idx = o_end * in_features
                embeddings_o_batch = embeddings_l[emb_start_idx:emb_end_idx] # (b*i, emb_d)      

                weights = metanet_l(embeddings_o_batch) 

                weights = weights.view(current_batch_o_size, in_features, self.d_b)  #(b,i,G+k+1)

                spline_weight = weights[:, :, :self.spline_dim].reshape(current_batch_o_size, -1).transpose(0, 1) # (i*(G+k), b)
                base_weight = weights[:, :, -1].transpose(0, 1) # (i, b)

                spline_output = torch.matmul(spline_basis, spline_weight) # (N, b)
                base_output = torch.matmul(base_feature, base_weight)

                layer_output[:, o_start:o_end] = base_output + spline_output # (N, b)
            
            target_shape = list(original_shape_prefix) + [out_features]
            current_x = layer_output.reshape(target_shape)
        
        return current_x

    def get_trainable_parameters(self):
        return list(self.embeddings.parameters()) + list(self.metanet.parameters())


