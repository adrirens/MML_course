import logging, os, sys, gc, time, re
from datetime import datetime
import torch, random, numpy
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, Subset
import matplotlib.pyplot as plt
import numpy as np

from fvcore.nn import FlopCountAnalysis, parameter_count

from torchvision.transforms import v2
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from models.mlp import *
from models.hyper import *
from models.kanbefair import *


from models.utils import *


from data.feynman import get_feynman_dataset


def get_timestamp():
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    print('Log directory: ', log_dir)
    return logger, formatter

def create_dataset(f, 
                   n_var=2, 
                   f_mode = 'col',
                   ranges = [-1,1],
                   train_num=1000, 
                   test_num=1000,
                   normalize_input=False,
                   normalize_label=False,
                   device='cpu',
                   seed=0):
    '''
    create dataset
    
    Args:
    -----
        f : function
            the symbolic formula used to create the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num : int
            the number of training samples. Default: 1000.
        test_num : int
            the number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            device. Default: 'cpu'.
        seed : int
            random seed. Default: 0.
        
    Returns:
    --------
        dataset : dic
            Train/test inputs/labels are dataset['train_input'], dataset['train_label'],
                        dataset['test_input'], dataset['test_label']
         
    Example
    -------
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = create_dataset(f, n_var=2, train_num=100)
    >>> dataset['train_input'].shape
    torch.Size([100, 2])
    '''

    np.random.seed(seed)
    torch.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var,2)
    else:
        ranges = np.array(ranges)
        
    
    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:,i] = torch.rand(train_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        test_input[:,i] = torch.rand(test_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
                
    if f_mode == 'col':
        train_label = f(train_input)
        test_label = f(test_input)
    elif f_mode == 'row':
        train_label = f(train_input.T)
        test_label = f(test_input.T)
    else:
        print(f'f_mode {f_mode} not recognized')
        
    # if has only 1 dimension
    if len(train_label.shape) == 1:
        train_label = train_label.unsqueeze(dim=1)
        test_label = test_label.unsqueeze(dim=1)
        
    def normalize(data, mean, std):
            return (data-mean)/std
            
    if normalize_input == True:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)
        
    if normalize_label == True:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)
    dataset = {}
    dataset['train_input'] = train_input.to(device)
    dataset['test_input'] = test_input.to(device)

    dataset['train_label'] = train_label.to(device)
    dataset['test_label'] = test_label.to(device)

    return dataset

def get_loader(args, shuffle = True, use_cuda = True):

    train_kwargs = {'batch_size': args.batch_size, 'num_workers': 0}
    test_kwargs = {'batch_size': args.test_batch_size, 'num_workers': 0}

    if shuffle:
        train_kwargs.update({'shuffle': True})
        test_kwargs.update({'shuffle': False})
    else:
        train_kwargs.update({'shuffle': False})
        test_kwargs.update({'shuffle': False})

    if args.dataset == "MNIST":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.MNIST('./dataset', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.MNIST('./dataset', train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_size = 1 * 28 * 28
    elif args.dataset in  [
        'test', 'I.6.20a', 'I.6.20', 'I.6.20b', 'I.8.4', 'I.9.18', 'I.10.7', 'I.11.19', 'I.12.1', 'I.12.2', 
        'I.12.4', 'I.12.5', 'I.12.11', 'I.13.4', 'I.13.12', 'I.14.3', 'I.14.4', 'I.15.3x', 'I.15.3t', 'I.15.10', 
        'I.16.6', 'I.18.4', 'I.18.4', 'I.18.16', 'I.24.6', 'I.25.13', 'I.26.2', 'I.27.6', 'I.29.4', 'I.29.16', 
        'I.30.3', 'I.30.5', 'I.32.5', 'I.32.17', 'I.34.8', 'I.34.10', 'I.34.14', 'I.34.27', 'I.37.4', 'I.38.12', 
        'I.39.10', 'I.39.11', 'I.39.22', 'I.40.1', 'I.41.16', 'I.43.16', 'I.43.31', 'I.43.43', 'I.44.4', 'I.47.23', 
        'I.48.20', 'I.50.26', 'II.2.42', 'II.3.24', 'II.4.23', 'II.6.11', 'II.6.15a', 'II.6.15b', 'II.8.7', 'II.8.31', 
        'I.10.9', 'II.11.3', 'II.11.17', 'II.11.20', 'II.11.27', 'II.11.28', 'II.13.17', 'II.13.23', 'II.13.34', 
        'II.15.4', 'II.15.5', 'II.21.32', 'II.24.17', 'II.27.16', 'II.27.18', 'II.34.2a', 'II.34.2', 'II.34.11', 
        'II.34.29a', 'II.34.29b', 'II.35.18', 'II.35.21', 'II.36.38', 'II.37.1', 'II.38.3', 'II.38.14', 'III.4.32', 
        'III.4.33', 'III.7.38', 'III.8.54', 'III.9.52', 'III.10.19', 'III.12.43', 'III.13.18', 'III.14.14', 'III.15.12', 
        'III.15.14', 'III.15.27', 'III.17.37', 'III.19.51', 'III.21.20'
    ]:
        symbol, expr, f, ranges, n_var = get_feynman_dataset(args.dataset)
        dataset = create_dataset(f, n_var=n_var,ranges=ranges, train_num=1000, test_num=1000,device=args.device)
        train_dataset = TensorDataset(dataset['train_input'], dataset['train_label'])
        test_dataset = TensorDataset(dataset['test_input'], dataset['test_label'])
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 1
        input_size = n_var


    else:
        raise NotImplementedError

    return train_loader, test_loader, num_classes, input_size


def get_continual_loader(args, shuffle = True, use_cuda = True):
    train_kwargs = {'batch_size': args.batch_size, 'num_workers': 4}
    test_kwargs = {'batch_size': args.test_batch_size, 'num_workers': 4}

    if shuffle:
        train_kwargs.update({'shuffle': True})
        test_kwargs.update({'shuffle': False})
    else:
        train_kwargs.update({'shuffle': False})
        test_kwargs.update({'shuffle': False})

    if args.dataset == "Class_MNIST":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.MNIST('./dataset', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.MNIST('./dataset', train=False, download=True,
                        transform=transform)
        train_labels = torch.Tensor(train_dataset.targets)
        test_labels = torch.Tensor(test_dataset.targets)
        train_index_dict = {}
        for i in range(10):
            train_index_dict[i] = (train_labels == i).nonzero().squeeze().tolist()
        test_index_dict = {}
        for i in range(10):
            test_index_dict[i] = (test_labels == i).nonzero().squeeze().tolist()
        train_datasets = [
            Subset(train_dataset, train_index_dict[0]+train_index_dict[1]+train_index_dict[2]),
            Subset(train_dataset, train_index_dict[3]+train_index_dict[4]+train_index_dict[5]),
            Subset(train_dataset, train_index_dict[6]+train_index_dict[7]+train_index_dict[8]),
        ]
        test_datasets = [
            Subset(test_dataset, test_index_dict[0]+test_index_dict[1]+test_index_dict[2]),
            Subset(test_dataset, test_index_dict[3]+test_index_dict[4]+test_index_dict[5]),
            Subset(test_dataset, test_index_dict[6]+test_index_dict[7]+test_index_dict[8]),
        ]
        train_loaders = [
            torch.utils.data.DataLoader(dataset,**train_kwargs) for dataset in train_datasets]
        test_loaders = [
            torch.utils.data.DataLoader(dataset, **test_kwargs) for dataset in test_datasets]
        num_classes = 9
        input_size = 1 * 28 * 28
    else:
        raise NotImplementedError

    return train_loaders, test_loaders, num_classes, input_size

def get_model(args):
    if args.model == "MLP":
        model = MLP(args)
    elif args.model == "KAN":
        model = KANbeFair(args)
    elif args.model == "HyperKAN":
        model = HyperKAN(args)

    else:
        raise NotImplementedError
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def randomness_control(seed):
    print("seed",seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_matrix(matrix, path):
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap='inferno')
    fig.colorbar(cax)
    fig.savefig(path)

def get_filename(path):
    base_name = os.path.basename(path)  # filename.extension
    file_name_without_extension = os.path.splitext(base_name)[0]
    return file_name_without_extension

def measure_time_memory(f):
    def wrapped(*args, **kwargs):
        if torch.cuda.is_available():
            start_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_max_memory_allocated()
        else:
            start_memory = 0

        start_time = time.time()

        result = f(*args, **kwargs)

        end_time = time.time()

        if torch.cuda.is_available():
            end_memory = torch.cuda.max_memory_allocated()
        else:
            end_memory = 0

        print(f"Function {f.__name__} executed in {end_time - start_time:.4f} seconds.")
        print(f"Memory usage increased by {(end_memory - start_memory) / (1024 ** 2):.2f} MB to {(end_memory) / (1024 ** 2):.2f} MB.")
        
        return result
    return wrapped

def classwise_validation(logits, label, targets, args):
    accuracies = []
    for target in targets:
        accuracies.append(accuracy = get_accuracy(logits, label, target))
    return accuracies

def get_accuracy(probability, label, target = None):
    prediction = probability.max(dim = 1)[1]
    if target is None:
        return ((prediction == label).sum() / label.numel()).item()
    else:
        mask = label == target
        return ((prediction[mask]== label[mask]).sum() / label[mask].numel()).item()

def get_activation(args):
    if args.activation_name == 'relu':
        return nn.ReLU
    elif args.activation_name == 'square_relu':
        return Square_ReLU
    elif args.activation_name == 'sigmoid':
        return nn.Sigmoid
    elif args.activation_name == 'tanh':
        return nn.Tanh
    elif args.activation_name == 'softmax':
        return nn.Softmax(dim=1)
    elif args.activation_name == 'silu':
        return nn.SiLU
    elif args.activation_name == 'gelu':
        return nn.GELU
    elif args.activation_name == 'glu':
        return nn.GLU
    elif args.activation_name == 'polynomial2':
        return Polynomial2
    elif args.activation_name == 'polynomial3':
        return Polynomial3
    elif args.activation_name == 'polynomial5':
        return Polynomial5
    else:
        raise ValueError(f'Unknown activation function: {args.activation_name}')

def get_shortcut_function(args):
    if args.kan_shortcut_name == 'silu':
        return nn.SiLU()
    elif args.kan_shortcut_name == 'identity':
        return nn.Identity()
    elif args.kan_shortcut_name == 'zero':

        class Zero(nn.Module):
            def __init__(self):
                super(Zero, self).__init__()
            def forward(self, x):
                return x * 0

        return Zero()
    else:
        raise ValueError(f'Unknown kan shortcut function: {args.kan_shortcut_name}')
    
def get_model_complexity(model, logger, args, method = "fvcore"):
    # using fvcore
    if method == "fvcore":
        parameter_dict = parameter_count(model)
        num_parameters = parameter_dict[""]

        # flops_dict = FlopCountAnalysis(model, torch.randn(2, args.input_size).to(args.device))
        # flops = flops_dict.total()
        flops = 0
    elif method == "coustomized":
        num_parameters = model.total_parameters()
        flops = model.total_flops()
    else:
        raise NotImplementedError

    if logger is not None:
        logger.info(f"Number of parameters: {num_parameters:,}; Number of FLOPs: {flops:,}")

    return num_parameters, flops

def write_results(args, subfix = "", **kwargs):
    result_base = "../results"
    result_file = f"results{subfix}.csv"

    dataset, model, general_parameters, specific_parameter = args.exp_id.split("/")[2:]
    general_parameters = general_parameters.split("__")
    specific_parameter = specific_parameter.split("__")

    result_file_path = os.path.join(result_base, result_file)
    
    s = [get_timestamp(), dataset, model] + general_parameters + specific_parameter + [str(kwargs[key]) for key in kwargs]
    s = ",".join(s) + "\n"
    if not os.path.exists(os.path.dirname(result_file_path)):
        os.makedirs(os.path.dirname(result_file_path))
    with open(result_file_path, "a") as f:
        f.write(s)

def todevice(obj, device):
    if isinstance(obj, (list,tuple)):
        obj = [o.to(device) for o in obj]
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    else:
        raise NotImplementedError
    return obj