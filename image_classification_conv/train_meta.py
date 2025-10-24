from __future__ import print_function

import json
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from fvcore.nn import FlopCountAnalysis, parameter_count



from tqdm import tqdm
from fvcore.common.timer import Timer
import numpy
import random
import logging
import warnings

import sys
from datetime import datetime
from models import  SimpleFastConvKAN, SimpleConvKAN,\
    EightSimpleFastConvKAN, EightSimpleConvKAN

from models.metafastconvkan import SimpleMetaFastConvKAN, EightFastMetaConvKAN, EightFastMetaConvKAN_L, EightFastMetaConvKAN_L3, EightFastMetaConvKAN_M
from models.metaconvkan import SimpleMetaConvKAN, EightSimpleMetaConvKAN, EightSimpleMetaConvKAN_L,EightSimpleMetaConvKAN_L3, EightSimpleMetaConvKAN_L4, EightSimpleMetaConvKAN_L5, EightSimpleMetaConvKAN_L6,\
EightSimpleMetaConvKAN_M
from models.metaconvkaln import SimpleMetaConvKALN, EightSimpleMetaConvKALN

from models.metaconvkagn import SimpleMetaConvKAGN, EightSimpleMetaConvKAGN, SimpleMetaConvKAGN_L4, SimpleMetaConvKAGN_L2,\
      EightSimpleMetaConvKAGN_L,EightSimpleMetaConvKAGN_L3, \
    EightSimpleMetaConvKAGN_L4, EightSimpleMetaConvKAGN_L2 
from models.metaconvkacn import SimpleMetaConvKACN, EightSimpleMetaConvKACN

warnings.simplefilter(action='ignore', category=UserWarning)

def get_timestamp():
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_time

def write_results(args, subfix = "", **kwargs):
    result_base = "../results"
    result_file = f"results{subfix}.csv"

    dataset, model, general_parameters = args.exp_id.split("/")[2:]
    general_parameters = general_parameters.split("__")
    # specific_parameter = specific_parameter.split("__")

    result_file_path = os.path.join(result_base, result_file)
    
    s = [get_timestamp(), dataset, model] + general_parameters + [str(kwargs[key]) for key in kwargs]
    s = ",".join(s) + "\n"
    if not os.path.exists(os.path.dirname(result_file_path)):
        os.makedirs(os.path.dirname(result_file_path))
    with open(result_file_path, "a") as f:
        f.write(s)


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

class OutputHook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)


def train_single(args, model_compiled, model, device, train_loader, optimizer, criterion,
          logger, output_hook, epoch, start_idx, scaler, l1_activation_penalty=0.0, l2_activation_penalty=0.0, is_moe=False):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader, start_idx):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model_compiled(data)
            moe_loss = 0
            if is_moe:
                output, moe_loss = output
            loss = criterion(output, target) + moe_loss

            l2_penalty = 0.
            l1_penalty = 0.
            for _output in output_hook:
                if l1_activation_penalty > 0:
                    l1_penalty += torch.norm(_output, 1, dim=0).mean()
                if l2_activation_penalty > 0:
                    l2_penalty += torch.norm(_output, 2, dim=0).mean()
            l2_penalty *= l2_activation_penalty
            l1_penalty *= l1_activation_penalty

            loss = loss + l1_penalty + l2_penalty
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % args.log_interval == 0:
            with torch.no_grad():
                output = model_compiled(data)
                losses = [criterion(output, target)]


            logger_info = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: '.format(
                epoch, (batch_idx - start_idx) * len(data), len(train_loader.dataset),
                100. * (batch_idx - start_idx) / len(train_loader)) + ",".join([str(l.item()) for l in losses])
            logger.info(logger_info)            

        output_hook.clear()
    return model


def train_double(args, model_compiled, model, device, train_loader, optimizer, optimH, optimE, criterion,
          logger, output_hook, epoch, start_idx, scaler, l1_activation_penalty=0.0, l2_activation_penalty=0.0, is_moe=False):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader, start_idx):

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            optimH.zero_grad()
            optimE.zero_grad()
            output = model_compiled(data)
            moe_loss = 0
            if is_moe:
                output, moe_loss = output
            loss = criterion(output, target) + moe_loss

            l2_penalty = 0.
            l1_penalty = 0.
            for _output in output_hook:
                if l1_activation_penalty > 0:
                    l1_penalty += torch.norm(_output, 1, dim=0).mean()
                if l2_activation_penalty > 0:
                    l2_penalty += torch.norm(_output, 2, dim=0).mean()
            l2_penalty *= l2_activation_penalty
            l1_penalty *= l1_activation_penalty

            loss = loss + l1_penalty + l2_penalty
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.step(optimH)
        scaler.step(optimE)
        scaler.update()

        if batch_idx % args.log_interval == 0:
            with torch.no_grad():
                output = model_compiled(data)
                losses = [criterion(output, target)]


            logger_info = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: '.format(
                epoch, (batch_idx - start_idx) * len(data), len(train_loader.dataset),
                100. * (batch_idx - start_idx) / len(train_loader)) + ",".join([str(l.item()) for l in losses])
            logger.info(logger_info)            

        output_hook.clear()
    return model

def test(model, device, test_loader, criterion, logger, name, is_moe=False):
    model.eval()
    test_loss = 0
    correct = 0    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if is_moe:
                output, _ = model(data, train=False)
            else:
                output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    logger.info("\t"+name+' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)    



    
def get_loaders(args):

    if args.dataset == 'MNIST':
        transform_train = v2.Compose([
            v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])

        transform_test = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
        # Load and transform the MNIST validation dataset
        valset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)
        # Create DataLoaders for training and validation datasets    
        input_channels = 1
        num_classes = 10    
    elif args.dataset == 'CIFAR10':
        transform_train = v2.Compose([

            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomChoice([v2.AutoAugment(AutoAugmentPolicy.CIFAR10),
                             v2.AutoAugment(AutoAugmentPolicy.IMAGENET),
                             v2.AutoAugment(AutoAugmentPolicy.SVHN),
                             v2.TrivialAugmentWide()]),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        transform_test = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True, transform=transform_train)
        # Load and transform the CIFAR10 validation dataset
        valset = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True, transform=transform_test)
        # Create DataLoaders for training and validation datasets
        input_channels = 3
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        transform_train = v2.Compose([

            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomChoice([v2.AutoAugment(AutoAugmentPolicy.CIFAR10),
                             v2.AutoAugment(AutoAugmentPolicy.IMAGENET),
                             v2.AutoAugment(AutoAugmentPolicy.SVHN),
                             v2.TrivialAugmentWide()]),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        transform_test = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.CIFAR100(root="../dataset", train=True, download=True, transform=transform_train)
        # Load and transform the CIFAR100 validation dataset
        valset = torchvision.datasets.CIFAR100(root="../dataset", train=False, download=True, transform=transform_test)
        # Create DataLoaders for training and validation datasets
        input_channels = 3
        num_classes = 100

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=12)
    valloader = DataLoader(valset, batch_size=args.test_batch_size, shuffle=False, num_workers=12)

    return trainloader, valloader, input_channels, num_classes

def get_model_complexity(model, logger, args):


    parameter_dict = parameter_count(model)
    num_parameters = parameter_dict[""]

    flops_dict = FlopCountAnalysis(model, torch.randn(2, args.input_channel, 32,32))
    flops = flops_dict.total()

    if logger is not None:
        logger.info(f"Number of parameters: {num_parameters:,}; Number of FLOPs: {flops:,}")

    return num_parameters, flops

def get_model(args, input_channels, num_classes):
    if args.norm_layer == 'batch':
        norm_layer = nn.BatchNorm2d
    elif args.norm_layer == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        norm_layer = None


    if args.model == 'MetaKAN':
        kan_model = SimpleMetaConvKAN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                             grid_size=5, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                             degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, norm_layer= nn.BatchNorm2d)
    elif args.model == 'MetaFastKAN':
        kan_model = SimpleMetaFastConvKAN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                             grid_size=8, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                             degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, norm_layer= nn.InstanceNorm2d)
    
    elif args.model == 'MetaKALN':
        kan_model = SimpleMetaConvKALN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                          degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                          degree_out=1,embedding_dim=args.embedding_dim, hidden_dim= args.hidden_dim,dropout_hyper=args.dropout_hyper, norm_layer= nn.InstanceNorm2d)
    elif args.model == 'MetaKAGN':
        kan_model =SimpleMetaConvKAGN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                          degree=3, groups=4, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                          degree_out=1,embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= norm_layer)
    elif args.model == 'MetaKAGN_L':
        kan_model =SimpleMetaConvKAGN_L4([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                          degree=3, groups=args.groups, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                          degree_out=1,embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= nn.InstanceNorm2d)        
 
    elif args.model == 'MetaKAGN_L2':
        kan_model =SimpleMetaConvKAGN_L2([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                          degree=3, groups=args.groups, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                          degree_out=1,embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= nn.InstanceNorm2d)     


    elif args.model == 'MetaKACN':
        kan_model = SimpleMetaConvKACN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                          degree=6, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                          degree_out=1,embedding_dim= args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= nn.BatchNorm2d)
    elif args.model == 'MetaKAN8':
        kan_model = EightSimpleMetaConvKAN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                  num_classes=num_classes, input_channels=input_channels,
                                  grid_size=5, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                  degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, norm_layer= norm_layer)
        
    elif args.model == 'MetaKAN8_M':
        kan_model = EightSimpleMetaConvKAN_M([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                  num_classes=num_classes, input_channels=input_channels, n_metanets = args.n_metanets,
                                  grid_size=5, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                  degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, norm_layer= norm_layer)        
       


    elif args.model == 'MetaFastKAN8':
        kan_model = EightFastMetaConvKAN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                  num_classes=num_classes, input_channels=input_channels,
                                  grid_size=8, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                  degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper = args.dropout_hyper, norm_layer= norm_layer)
        


    elif args.model == 'MetaKALN8':
        kan_model = EightSimpleMetaConvKALN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                               num_classes=num_classes, input_channels=input_channels,
                               degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                               degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= nn.InstanceNorm2d)

    elif args.model == 'MetaKACN8':
        kan_model = EightSimpleMetaConvKACN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                               num_classes=num_classes, input_channels=input_channels,
                               degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                               degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= nn.BatchNorm2d)

    elif args.model == 'MetaFastKAN8_L':
        kan_model = EightFastMetaConvKAN_L([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                  num_classes=num_classes, input_channels=input_channels,
                                  grid_size=8, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                  degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper = args.dropout_hyper, norm_layer= nn.InstanceNorm2d)

    elif args.model == 'MetaFastKAN8_M':
        kan_model = EightFastMetaConvKAN_M([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                  num_classes=num_classes, input_channels=input_channels,n_metanets = args.n_metanets,
                                  grid_size=8, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                  degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper = args.dropout_hyper, norm_layer= norm_layer)


    elif args.model == 'MetaFastKAN8_L3':
        kan_model = EightFastMetaConvKAN_L3([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                  num_classes=num_classes, input_channels=input_channels,
                                  grid_size=8, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                  degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper = args.dropout_hyper, norm_layer= norm_layer)


    elif args.model == 'MetaKAN8_L':
        kan_model = EightSimpleMetaConvKAN_L([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                  num_classes=num_classes, input_channels=input_channels,
                                  grid_size=5, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                  degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= nn.InstanceNorm2d)

    elif args.model == 'MetaKAN8_L3':
        kan_model = EightSimpleMetaConvKAN_L3([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                  num_classes=num_classes, input_channels=input_channels,
                                  grid_size=5, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                  degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= nn.InstanceNorm2d)

    elif args.model == 'MetaKAN8_L4':
        kan_model = EightSimpleMetaConvKAN_L4([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                  num_classes=num_classes, input_channels=input_channels,
                                  grid_size=5, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                  degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= norm_layer)

    elif args.model == 'MetaKAN8_L5':
        kan_model = EightSimpleMetaConvKAN_L5([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                  num_classes=num_classes, input_channels=input_channels,
                                  grid_size=5, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                  degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= nn.InstanceNorm2d)


    elif args.model == 'MetaKAN8_L6':
        kan_model = EightSimpleMetaConvKAN_L6([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                  num_classes=num_classes, input_channels=input_channels,
                                  grid_size=5, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                  degree_out=1, embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= nn.InstanceNorm2d)


    elif args.model == 'MetaKAGN8':
        kan_model = EightSimpleMetaConvKAGN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                        num_classes=num_classes, input_channels=input_channels,
                        degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                        degree_out=1)
                        
    elif args.model == 'MetaKAGN8_L':
        kan_model = EightSimpleMetaConvKAGN_L([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                        num_classes=num_classes, input_channels=input_channels,
                        degree=3, groups=args.groups, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                        degree_out=1,embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= nn.InstanceNorm2d)                        


    elif args.model == 'MetaKAGN8_L3':
        kan_model = EightSimpleMetaConvKAGN_L3([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                        num_classes=num_classes, input_channels=input_channels,
                        degree=3, groups=args.groups, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                        degree_out=1,embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= norm_layer)   




    elif args.model == 'MetaKAGN8_L4':
        kan_model = EightSimpleMetaConvKAGN_L4([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                        num_classes=num_classes, input_channels=input_channels,
                        degree=3, groups=args.groups, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                        degree_out=1,embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= nn.InstanceNorm2d)     


    elif args.model == 'MetaKAGN8_L2':
        kan_model = EightSimpleMetaConvKAGN_L2([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                        num_classes=num_classes, input_channels=input_channels,
                        degree=3, groups=args.groups, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                        degree_out=1,embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout_hyper=args.dropout_hyper, norm_layer= nn.InstanceNorm2d)                
    return kan_model

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Training')    
    parser.add_argument('--model', type=str, default="KAN", #required=True,
                        help='network structure')    
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='supported optimizer: adam, lbfgs')
    parser.add_argument('--scheduler', type=str, default="exponential",
                    help='scheduler of optimizer: adam, lbfgs')    
    parser.add_argument('--dataset', type=str, default="CIFAR10", #required=True,
                        help='dataset')    
    parser.add_argument('--groups', type=int, default=1,
                        help='number of groups for group convolution')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id to use (default: 0)') 
    
    parser.add_argument('--norm_layer', type=str, default="batch",
                        help='norm layer: batch, instance')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=150, # 100 MNIST pretrain, 5 Finetune
                        help='number of epochs to train (default: 14)')    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--dropout_hyper',type=float, default= 0,
                        help="dropout for the hypernet")
    parser.add_argument('--gamma', type=float, default=0.975,
                        help='Learning rate step gamma (default: 0.7, 1.0 for fewshot)')    
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--wd_e', type=float, default=1e-5,
                        help='weight decay (default: 0.0)')    
    
    parser.add_argument('--optim_set', type=str, default="double",
                        help='optimizer settings for hypernet, option: single  double')    
    parser.add_argument('--seed', type=int, default=1314,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-model-interval', type = int, default=-1, 
                        help='whether save model along training')    
    
    ################# Parameters for hypernetwork #################
    parser.add_argument('--embedding_dim', type=int, default=1, 
                        help='dimension of embedding vector') 
    parser.add_argument('--n_metanets', type=int, default=1,
                        help='number of metanets for MetaKAN8_M and MetaFastKAN8_M')
    parser.add_argument('--hidden_dim',type=int, default= 64,
                        help='hidden dimension of hypernet')  
    parser.add_argument('--lr_h',type=float, default= 1e-3,
                        help='learning rate of hypernet')
    parser.add_argument('--lr_e',type=float, default= 1e-2,
                        help='learing rate of embedding')        
    parser.add_argument('--emb_sched', action='store_true', default=False,
                    help='scheduler of embedding')
    parser.add_argument('--hypernet_sched', action='store_true',default=False,
                        help='scheduler of hypernetwork')
    
    parser.add_argument('--layer_emb_dim', type=int, default=1,
                        help='dimension of embedding vector for each layer')
    ################# Parameters for hypernetwork #################
    


    args = parser.parse_args()

    use_cuda =  torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")    

    randomness_control(args.seed)
    args.device = device
    args.save_model_along = args.save_model_interval > 0        

    args.exp_id = f"./logs/{args.dataset}/{args.model}/"    
    args.exp_id = args.exp_id + f"{args.embedding_dim}__{args.hidden_dim}__{args.batch_size}__{args.epochs}__{args.seed}__{args.gamma}__{args.dropout_hyper}__{args.scheduler}__{args.groups}__{args.norm_layer}"
    if args.model in ['MetaKAN8_M','MetaFastKAN8_M']:
        args.exp_id = args.exp_id + f"__{args.n_metanets}"
    if args.optim_set == 'double':
        args.exp_id = args.exp_id + f"__{args.optim_set}__{args.lr_h}__{args.lr_e}__{args.wd_e}__{args.wd}__{args.emb_sched}__{args.hypernet_sched}"
    elif args.optim_set == 'single':
        args.exp_id = args.exp_id + f"__{args.optim_set}__{args.lr}__{args.wd}"    
    
    if args.model in  ['MetaKAN8_DEL','MetaFastKAN8_DEL_3','MetaKAGN8_DE']:
        args.exp_id = args.exp_id + f"__{args.layer_emb_dim}"
    
    os.makedirs(args.exp_id, exist_ok=True)

    if os.path.exists(os.path.join(args.exp_id, "log")):
        with open(os.path.join(args.exp_id, "log"), "r") as f:
            lines = f.readlines()
            if len(lines) > 0:
                if "training process was finished" in lines[-1]:
                    raise ValueError("training process was finished")
    
    logger, formatter = get_logger(args.exp_id, None, "log", level=logging.INFO)


    train_loader, val_loader, input_channels, num_classes = get_loaders(args)

    args.input_channel = input_channels
    args.num_classes = num_classes

    model = get_model(args, input_channels, num_classes)

    logger.info(model)
    num_parameters, flops = get_model_complexity(model, logger, args)

    model.to(device)

    
    hypernet = model.hyper_net
    embedding = model.embeddings

    if args.optim_set == 'double':
        optimizer = optim.AdamW(
            [param for name, param in model.named_parameters() if "hyper_net" not in name and "embeddings" not in name],
            lr=args.lr, weight_decay=args.wd
        )    
        optimH = optim.AdamW(hypernet.parameters(), lr=args.lr_h, weight_decay=args.wd)
        optimE = optim.AdamW(embedding.parameters(), lr=args.lr_e, weight_decay=args.wd_e)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        if args.scheduler == 'exponential':
            
            scheduler_h = optim.lr_scheduler.ExponentialLR(optimH, gamma=args.gamma)
            scheduler_e =  optim.lr_scheduler.ExponentialLR(optimizer = optimE,
                                                                gamma=args.gamma) 
        elif args.scheduler =='cos':

            scheduler_h =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimH,
                                                                T_max =  -1) #  * iters 
            scheduler_e =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimE,

                                                                T_max =  -1) #  * iters 
    elif args.optim_set =='single':
        if args.optimizer == "adam":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
        elif args.optimizer =='adamw':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        else:
            raise NotImplementedError
        if args.scheduler == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

        elif args.scheduler =='cos':
            scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                                T_max =  -1) #  * iters 
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    output_hook = OutputHook()
    fvctimer = Timer()

    best_test_metric = 0 
    corresponding_train_metric = 0

    for epoch in range(1, args.epochs + 1):
        if fvctimer.is_paused():
            fvctimer.resume()
        else:
            fvctimer.reset()        
        if args.optim_set == 'double':

            model = train_double(args, model, model, device, train_loader, optimizer, optimH, optimE, criterion, logger, output_hook, epoch, start_idx=(epoch-1)*len(train_loader), scaler=scaler)
        elif args.optim_set == 'single':
            model = train_single(args, model, model, device, train_loader, optimizer, criterion, logger, output_hook, epoch, start_idx=(epoch-1)*len(train_loader), scaler=scaler)
        
        fvctimer.pause()
        train_metric = test(model, device, train_loader, criterion= criterion, logger = logger, name = "train")
        test_metric = test( model, device, val_loader, criterion= criterion, logger = logger, name = "test")

        if test_metric > best_test_metric:
            best_test_metric = test_metric
            corresponding_train_metric = train_metric

        scheduler.step()
        if args.emb_sched:
            scheduler_e.step()
        if args.hypernet_sched:
            scheduler_h.step()


    total_training_time = fvctimer.seconds()
    average_training_time_per_epoch = fvctimer.avg_seconds()
    logger.info(f"total training time: {total_training_time:,} seconds; average training time per epoch: {average_training_time_per_epoch:,} seconds")

    write_results(
        args,
        train_metric = corresponding_train_metric,
        test_metric = best_test_metric,
        num_parameters = num_parameters,
        flops = flops,
        total_training_time = total_training_time,
        average_training_time_per_epoch = average_training_time_per_epoch
    )    

    logger.info(f"training process was finished")


if __name__ == '__main__':
    main()