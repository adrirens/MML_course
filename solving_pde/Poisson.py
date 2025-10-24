import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
from models.kan import KAN
from models import kan, metakan
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PINN Training')
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--dim', type=int, default=4) # dimension of the problem.
parser.add_argument('--dataset', type=str, default="Poisson")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--epochs', type=int, default=10000) # Adam epochs
parser.add_argument('--lr', type=float, default=1e-3) # Adam lr
parser.add_argument('--lr_h', type=float, default=1e-3)
parser.add_argument('--lr_e', type=float, default=1e-2)
parser.add_argument('--PINN_h', type=int, default=128) # width of PINN
parser.add_argument('--PINN_L', type=int, default=4) # depth of PINN
parser.add_argument('--save_loss', type=bool, default=True) # save the optimization trajectory?
parser.add_argument('--use_sch', type=int, default=1) # use scheduler?
parser.add_argument('--N_f', type=int, default=int(100)) # num of residual points
parser.add_argument('--N_test', type=int, default=int(20000)) # num of test points
parser.add_argument('--x_radius', type=float, default=1)
parser.add_argument('--method', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--model', type=str, default='HyperKAN')
parser.add_argument('--grid', type=int, default=5)
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--embedding_dim', type=int, default=1)
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()
print(args)

# device = torch.device(args.device)
device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
np.random.seed(args.SEED)
assert args.dataset == "Poisson"

c = np.random.randn(1, args.dim - 1)
const_2 = 1
def load_data_TwoBody_Poisson(d):
    global func_u
    args.input_dim = d
    args.output_dim = 1
    def func_u(x):
        temp =  args.x_radius**2 - np.sum(x**2, 1)
        temp2 = c * np.sin(x[:, :-1] + const_2 * np.cos(x[:, 1:]) + x[:, 1:] * np.cos(x[:, :-1]))
        temp2 = np.sum(temp2, 1)
        return temp * temp2

    N_test = args.N_test

    x = np.random.randn(N_test, d)
    r = np.random.rand(N_test, 1) * args.x_radius
    x = x / np.linalg.norm(x, axis=1, keepdims=True) * r
    u = func_u(x)
    return x, u
x, u = load_data_TwoBody_Poisson(d=args.dim)
func_u = func_u
x, u = load_data_TwoBody_Poisson(d=args.dim)
print(x.shape, u.shape)
print(u.mean(), u.std())

class MLP(nn.Module):
    def __init__(self, layers:list):
        super(MLP, self).__init__()
        models = []
        for i in range(len(layers)-1):
            models.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                models.append(nn.Tanh())
        self.nn = nn.Sequential(*models)
    def forward(self, x):
        return ((1 - torch.sum(x**2, 1, keepdims=True)) * self.nn(x))

class KAN(nn.Module):
    def __init__(self, layers:list, grid, k):
        super(KAN, self).__init__()
        self.kan = KAN(width = layers, grid = grid, k = k)

    def forward(self, x):
        return ((1 - torch.sum(x**2, 1, keepdims=True)) * self.kan(x))
    
class MetaKAN(nn.Module):
    def __init__(self, layers:list, grid, k, hidden_dim, embedding_dim, device):  
        super(MetaKAN, self).__init__()
        self.hyperkan = metakan.MetaKAN(layers_hidden = layers, grid_size = grid, spline_order = k, hidden_dim = hidden_dim, embedding_dim = embedding_dim)
        self.embeddings = self.hyperkan.embeddings
        self.metanet = self.hyperkan.hypernet

    def forward(self, x):
        return ((1 - torch.sum(x**2, 1, keepdims=True)) * self.hyperkan(x)) 



class PINN:
    def __init__(self):
        self.epoch = args.epochs
        self.adam_lr = args.lr
        self.lr_e = args.lr_e
        self.lr_h = args.lr_h
        self.dim, self.batch_size = args.dim, args.batch_size
        self.x = torch.tensor(x, dtype=torch.float32, requires_grad=False).to(device)
        self.u = torch.tensor(u, dtype=torch.float32, requires_grad=False).to(device).reshape(-1)
        # Initalize Neural Networks
        layers = [args.input_dim] + [args.PINN_h] * (args.PINN_L - 1) + [args.output_dim]
        if args.model == 'MLP':
            self.u_net = MLP(layers).to(device)
        elif args.model == 'KAN':
            self.u_net = KAN(layers, grid = args.grid, k=args.k).to(device)
        elif args.model == 'MetaKAN':
            self.u_net = MetaKAN(layers, grid=args.grid, k=args.k, 
                                 hidden_dim=args.hidden_dim, 
                                 embedding_dim=args.embedding_dim, 
                                 device=device).to(device)
            
            params_embeddings = list(self.u_net.embeddings.parameters())
            params_hypernet = list(self.u_net.metanet.parameters())

            self.optimizer_emb = torch.optim.Adam(params_embeddings, lr=self.lr_e)
            print(f"MetaKAN: Initialized optimizer for embeddings with {sum(p.numel() for p in params_embeddings)} parameters.")

            self.optimizer_hyper = torch.optim.Adam(params_hypernet, lr=self.lr_h)
            print(f"MetaKAN: Initialized optimizer for hypernet with {sum(p.numel() for p in params_hypernet)} parameters.")
        # self.u_net = MLP(layers).to(device)
        print(self.u_net)
        self.net_params_pinn = list(self.u_net.parameters())
        self.saved_loss = []
        self.saved_l2 = []

    def Resample(self): # sample random points at the begining of each iteration
        N_f = args.N_f # Number of collocation points

        xf = np.random.randn(N_f, args.dim)
        rf = np.random.rand(N_f, 1) * args.x_radius
        xf = xf / np.linalg.norm(xf, axis=1, keepdims=True) * rf
        x = xf

        u1 = args.x_radius**2 - np.sum(x**2, 1, keepdims=True)
        du1_dx = -2 * x
        d2u1_dx2 = -2

        x1, x2 = x[:, :-1], x[:, 1:]
        coeffs = c
        u2 = coeffs * np.sin(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1))
        u2 = np.sum(u2, 1, keepdims=True)
        du2_dx_part1 = coeffs * np.cos(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * \
                (1 - x2 * np.sin(x1))
        du2_dx_part2 = coeffs * np.cos(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * \
                (-const_2 * np.sin(x2) + np.cos(x1))
        du2_dx = np.zeros((N_f, args.dim))
        du2_dx[:, :-1] += du2_dx_part1
        du2_dx[:, 1:] += du2_dx_part2
        d2u2_dx2_part1 = -coeffs * np.sin(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * (1 - x2 * np.sin(x1))**2 + \
                coeffs * np.cos(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * (- x2 * np.cos(x1))
        d2u2_dx2_part2 = -coeffs * np.sin(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * (-const_2 * np.sin(x2) + np.cos(x1))**2 + \
                coeffs * np.cos(x1 + const_2 * np.cos(x2) + x2 * np.cos(x1)) * (-const_2 * np.cos(x2))
        d2u2_dx2 = np.zeros((N_f, args.dim))
        d2u2_dx2[:, :-1] += d2u2_dx2_part1
        d2u2_dx2[:, 1:] += d2u2_dx2_part2
        ff = u1 * d2u2_dx2 + 2 * du1_dx * du2_dx + u2 * d2u1_dx2
        ff = np.sum(ff, 1)

        self.xf = torch.tensor(xf, dtype=torch.float32, requires_grad=True).to(device)
        self.ff = torch.tensor(ff, dtype=torch.float32, requires_grad=True).to(device)
        return

    def Method0(self): # Vanilla PINN
        x = self.xf
        x.requires_grad_()
        f = self.u_net(x)
        u_x = torch.autograd.grad(f.sum(), x, create_graph=True)[0]

        u_xx = []
        for i in range(self.dim):
            d2f_dxidxi = torch.autograd.grad(u_x[:, i].sum(), x, create_graph=True)[0][:, i:i+1]
            u_xx.append(d2f_dxidxi)
        # (batch_size, x_dim)
        u_xx = torch.concat(u_xx, dim=1)

        residual_pred = torch.sum(u_xx, dim=1) - self.ff
        loss = residual_pred.square().mean()
        saeved_loss = loss
        return loss, saeved_loss
    
    def Method3(self): #SDGD Algorithm 3
        x = self.xf
        x.requires_grad_()
        f = self.u_net(x)
        u_x = torch.autograd.grad(f.sum(), x, create_graph=True)[0]

        residual_pred = 0
        idx = np.random.choice(self.dim, self.batch_size, replace=False)
        for i in idx:
            d2f_dxidxi = torch.autograd.grad(u_x[:, i].sum(), x, create_graph=True)[0][:, i]
            residual_pred += d2f_dxidxi

        residual_pred = residual_pred * self.dim / self.batch_size - self.ff
        loss = residual_pred.square().mean()
        saeved_loss = loss
        return loss, saeved_loss

    def num_params(self):
        num_pinn = 0
        for p in self.net_params_pinn:
            num_pinn += len(p.reshape(-1))
        return num_pinn

    def train_adam(self):
        optimizer = torch.optim.Adam(self.net_params_pinn, lr=self.adam_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
        lr_lambda = lambda epoch: 1-epoch/args.epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        L2, L1 = self.L2_pinn()
        print('Initialization: l2: %e, l1: %e'%(L2, L1))
        self.saved_loss.append(0)
        self.saved_l2.append([L2, L1])
    
        for n in tqdm(range(self.epoch)):
            self.Resample()
            if args.method == 0:
                loss, saved_loss = self.Method0()
            elif args.method == 3:
                loss, saved_loss = self.Method3()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.use_sch:
                scheduler.step()
            current_loss = saved_loss.item()
            if n % 100 == 0:
                L2, L1 = self.L2_pinn()
                print('epoch %d, loss: %e, l2: %e, l1: %e'%(n, current_loss, L2, L1))
            if args.save_loss:
                self.saved_loss.append(current_loss)
                if n % 100 != 0:
                    L2, L1 = self.L2_pinn()
                self.saved_l2.append([L2, L1])

    def predict_pinn(self):
        f = self.u_net(self.x).reshape(-1)
        return f
    
    def L2_pinn(self):
        pred_u = self.predict_pinn()
        pred_u = self.u - pred_u
        L2, L1 = torch.norm(pred_u) / torch.norm(self.u), \
            torch.norm(pred_u, p=1) / torch.norm(self.u, 1)
        L2, L1 = L2.item(), L1.item()
        return L2, L1






model = PINN()
print("Num params:", model.num_params())
model.train_adam()
print(model.u_net)

if args.save_loss:
    model.saved_loss = np.asarray(model.saved_loss)
    model.saved_l2 = np.asarray(model.saved_l2)
    info_dict = {"loss": model.saved_loss, "L2": model.saved_l2[:, 0], "L1": model.saved_l2[:, 1]}
    df = pd.DataFrame(data=info_dict, index=None)
    df.to_excel(
        "saved_loss_l2/"+args.dataset+"_dim="+str(args.dim)+\
            "_batch="+str(args.batch_size)+"_N_f="+str(args.N_f)\
            +"_method="+str(args.method)+"_SEED="+str(args.SEED)+".xlsx",
        index=False
    )

