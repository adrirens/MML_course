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
from models import  SimpleFastConvKAN, SimpleConvKAN, SimpleConv, \
    EightSimpleFastConvKAN, EightSimpleConvKAN, EightSimpleConv, SimpleConvWavKAN, EightSimpleConvWavKAN,SimpleConvKACN, EightSimpleConvKACN,SimpleConvKAGN, EightSimpleConvKAGN,\
    SimpleConvKALN, EightSimpleConvKALN



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


def log_memory_usage(device, stage=""):
    """记录并打印当前和峰值显存使用情况 (单位：MB)"""
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    print(f"[{stage:<25}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | "\
          f"Peak Allocated: {max_allocated:.2f} MB | Peak Reserved: {max_reserved:.2f} MB")

def train(args, model_compiled, model, device, train_loader, optimizer, criterion,
          logger, output_hook, epoch, start_idx, scaler, l1_activation_penalty=0.0, l2_activation_penalty=0.0, is_moe=False):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader, start_idx):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            torch.cuda.reset_peak_memory_stats(device)
            log_memory_usage(args.device, "Step Start (Reset Peak)")                    
            data, target = data.to(device), target.to(device)
            log_memory_usage(args.device,"After Data to GPU")

            optimizer.zero_grad()
            output = model_compiled(data)
            log_memory_usage(args.device,"After Forward Pass")
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
        log_memory_usage(args.device, "After Loss Calculation") # loss 通常很小
        scaler.step(optimizer)
        log_memory_usage(args.device, "After optimizer.step()") # 优化器更新内部状态
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
        trainset = torchvision.datasets.MNIST(root="/home/ubuntu/Desktop/zzc/scaling/dataset", train=True, download=True, transform=transform_train)
        # Load and transform the MNIST validation dataset
        valset = torchvision.datasets.MNIST(root="/home/ubuntu/Desktop/zzc/scaling/dataset", train=False, download=True, transform=transform_test)
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
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        # Load and transform the CIFAR10 validation dataset
        valset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
        # Create DataLoaders for training and validation datasets
        input_channels = 3
        num_classes = 10
    else:
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
        trainset = torchvision.datasets.CIFAR100(root="/home/ubuntu/Desktop/zzc/scaling/dataset", train=True, download=True, transform=transform_train)
        # Load and transform the CIFAR100 validation dataset
        valset = torchvision.datasets.CIFAR100(root="/home/ubuntu/Desktop/zzc/scaling/dataset", train=False, download=True, transform=transform_test)
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

    if args.model == 'KAN':
        kan_model = SimpleConvKAN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                         spline_order=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                         degree_out=1)

    elif args.model == 'KAN8':
        kan_model = EightSimpleConvKAN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                              num_classes=num_classes, input_channels=input_channels,
                              spline_order=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.000000,
                              degree_out=1)       
    elif args.model == 'FastKAN':
        kan_model = SimpleFastConvKAN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                             grid_size=8, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                             degree_out=1, norm_layer=nn.InstanceNorm2d) 
    elif args.model == 'FastKAN8':
        kan_model = EightSimpleFastConvKAN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                  num_classes=num_classes, input_channels=input_channels,
                                  grid_size=8, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                  degree_out=1,norm_layer=nn.InstanceNorm2d)
    elif args.model == 'KALN':
        kan_model = SimpleConvKALN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                          degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                          degree_out=1)
    elif args.model == 'KAGN':
        kan_model =SimpleConvKAGN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                          degree=3, groups=args.groups, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                          degree_out=1)
        
    elif args.model == 'KACN':
        kan_model = SimpleConvKACN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                          degree=6, groups=4, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                          degree_out=1)
    elif args.model == 'WavKAN':
        kan_model = SimpleConvWavKAN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=num_classes, input_channels=input_channels,
                            wavelet_type='mexican_hat', groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                            degree_out=1)

    elif args.model == 'KALN8':
        kan_model = EightSimpleConvKALN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                               num_classes=num_classes, input_channels=input_channels,
                               degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                               degree_out=1)
        
    elif args.model == 'KAGN8':
        kan_model = EightSimpleConvKAGN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                               num_classes=num_classes, input_channels=input_channels,
                               degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                               degree_out=1, norm_layer=norm_layer)
        
    elif args.model == 'KACN8':
        kan_model = EightSimpleConvKACN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                               num_classes=num_classes, input_channels=input_channels,
                               degree=3, groups=1, dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                               degree_out=1)
    
    elif args.model == 'WavKAN8':
        kan_model = EightSimpleConvWavKAN([8 * 2, 16 * 2, 32 * 2, 64 * 2, 128 * 2, 128 * 2, 128 * 4, 128 * 4],
                                 num_classes=num_classes, input_channels=input_channels,
                                 wavelet_type='mexican_hat', groups=1,
                                 dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000,
                                 degree_out=1)

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
    parser.add_argument('--norm_layer', type=str, default="batch",
                        help='norm layer')
    parser.add_argument('--groups', type=int, default=1,
                        help='number of groups for group convolution')    
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id to use (default: 0)') 
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=150, # 100 MNIST pretrain, 5 Finetune
                        help='number of epochs to train (default: 14)')    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.975,
                        help='Learning rate step gamma (default: 0.7, 1.0 for fewshot)')    
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='weight decay (default: 0.0)')
    
    parser.add_argument('--seed', type=int, default=1314,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--save-model-interval', type = int, default=-1, 
                        help='whether save model along training')    
    

    


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
    args.exp_id = args.exp_id + f"{args.batch_size}__{args.epochs}__{args.lr}__{args.wd}__{args.seed}__{args.gamma}__{args.norm_layer}"
    
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
    print("\n--- Memory Before Training ---")
    log_memory_usage(device=args.device,stage="Initial state")


    model = get_model(args, input_channels, num_classes)

    logger.info(model)
    num_parameters, flops = get_model_complexity(model, logger, args)

    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    output_hook = OutputHook()
    fvctimer = Timer()

    # 模型参数和优化器状态会占用初始内存
    log_memory_usage(device=args.device, stage="After Model & Optimizer")

    best_test_metric = 0 
    corresponding_train_metric = 0
    print("\n--- Starting Training Loop ---")
    # 重置整个训练过程的峰值统计 (可选)
    torch.cuda.reset_peak_memory_stats(args.device)
    log_memory_usage(device=args.device, stage="After reset peak stats")    
    for epoch in range(1, args.epochs + 1):
        if fvctimer.is_paused():
            fvctimer.resume()
        else:
            fvctimer.reset()        

        model = train(args, model, model, device, train_loader, optimizer, criterion, logger, output_hook, epoch, start_idx=(epoch-1)*len(train_loader), scaler=scaler)
        fvctimer.pause()
        train_metric = test(model, device, train_loader, criterion= criterion, logger = logger, name = "train")
        test_metric = test( model, device, val_loader, criterion= criterion, logger = logger, name = "test")

        if test_metric > best_test_metric:
            best_test_metric = test_metric
            corresponding_train_metric = train_metric

        scheduler.step()


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