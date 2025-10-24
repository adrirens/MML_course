import logging, os, sys, gc, time, re
from datetime import datetime
import torch, random, numpy
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, Subset
import matplotlib.pyplot as plt
from fvcore.nn import  parameter_count


from models.kan import KAN
from models.mlp import MLP
from models.metakan import MetaKAN
from models.fastkan import FastKAN
from models.wavkan import WavKAN
from models.metafastkan import MetaFastKAN
from models.metawavkan import MetaWavKAN


def get_model_complexity(model, logger, args, method = "fvcore"):

    parameter_dict = parameter_count(model)
    num_parameters = parameter_dict[""]


    if logger is not None:
        logger.info(f"Number of parameters: {num_parameters:}")

    return num_parameters

def write_results(args, subfix = "", **kwargs):
    result_base = "./results"
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

def randomness_control(seed):
    print("seed",seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        train_dataset = datasets.MNIST('../dataset', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.MNIST('../dataset', train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_size = 1 * 28 * 28

    elif args.dataset == "EMNIST-Letters":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1724,), (0.3311,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.EMNIST('../dataset', split = "letters", train=True, download=True,
                        transform=transform)
        test_dataset = datasets.EMNIST('../dataset', split = "letters", train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 37
        input_size = 1 * 28 * 28
 
    elif args.dataset == "EMNIST-Balanced":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1753,), (0.3334,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.EMNIST('../dataset', split = "balanced",  train=True, download=True,
                        transform=transform)
        test_dataset = datasets.EMNIST('../dataset', split = "balanced",  train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 47
        input_size = 1 * 28 * 28

    elif args.dataset == "FMNIST":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.FashionMNIST('../dataset', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.FashionMNIST('../dataset', train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_size = 1 * 28 * 28

    elif args.dataset == "KMNIST":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1918,), (0.3483,)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.KMNIST('../dataset', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.KMNIST('../dataset', train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_size = 1 * 28 * 28

    elif args.dataset == "Cifar10":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Resize(28),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.CIFAR10('../dataset', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.CIFAR10('../dataset', train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_size = 3 * 28 * 28

    elif args.dataset == "Cifar100":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Resize(28),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.CIFAR100('../dataset', train=True, download=True,
                        transform=transform)
        test_dataset = datasets.CIFAR100('../dataset', train=False, download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 100
        input_size = 3 * 28 * 28        


    elif args.dataset == "SVHN":
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.4381, 0.4442, 0.4734), (0.1983, 0.2013, 0.1972)),
            transforms.Lambda(lambda x: torch.flatten(x))
            ])
        train_dataset = datasets.SVHN('../dataset', split ="train", download=True,
                        transform=transform)
        test_dataset = datasets.SVHN('../dataset', split ="test", download=True,
                        transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
        num_classes = 10
        input_size = 3 * 28 * 28
                
    else:
        raise NotImplementedError

    return train_loader, test_loader, num_classes, input_size                

def get_activation(args):
    if args.base_activation == 'silu':
        return nn.SiLU()
    elif args.base_activation == 'identity':
        return nn.Identity()
    elif args.base_activation == 'zero':

        class Zero(nn.Module):
            def __init__(self):
                super(Zero, self).__init__()
            def forward(self, x):
                return x * 0

        return Zero()
    else:
        raise ValueError(f'Unknown kan shortcut function: {args.base_activation}')
    
def get_model(args):
    if args.model == "MLP":
        model = MLP(args)
    elif args.model == "KAN":
        model = KAN(args)
    elif args.model == "MetaKAN":
        model = MetaKAN(args)
    elif args.model == "FastKAN":
        model = FastKAN(args)
    elif args.model == "WavKAN":
        model = WavKAN(args)   
    elif args.model == "HyperFastKAN":
        model = MetaFastKAN(args)      
    elif args.model == "HyperWavKAN":
        model = MetaWavKAN(args)      
    else:
        raise NotImplementedError
    return model    