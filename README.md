# MetaKAN
Official Code Implementation of "Improving Memory Efficiency for Training KANs via Meta Learning"  
📄 [Paper Link (arXiv:2506.07549)](https://arxiv.org/pdf/2506.07549)


This paper proposes a new method called **MetaKAN**, which uses meta-learning strategies to significantly reduce the memory consumption of Kolmogorov-Arnold Networks (KANs) during training while maintaining their powerful performance.

## Table of Contents

- [Project Overview](#project-overview)
- [Project File Structure](#project-file-structure)
- [Quick Start: Running Experiments](#quick-start-running-experiments)
  - [Function Fitting](#function-fitting)
  - [Image Classification (Fully Connected)](#image-classification-fully-connected)
  - [Image Classification (Convolutional)](#image-classification-convolutional)
  - [Solving Partial Differential Equations (PDEs)](#solving-partial-differential-equations-pdes)
- [How to Cite](#how-to-cite)



## Project Overview

Kolmogorov-Arnold Networks (KANs) are a promising alternative to neural networks, demonstrating great potential in fields such as mathematical reasoning and scientific computing. However, their high memory consumption has been a major practical bottleneck. This project addresses this issue by introducing a meta-learning framework that dynamically generates the spline function parameters of KANs during training. Our method demonstrates performance comparable to or even superior to the original KAN in multiple benchmark tests, including function fitting, image classification, and solving partial differential equations, while achieving significant improvements in memory efficiency.


## Project File Structure

Below is the core file structure of this project and a brief introduction to each section:

```
MetaKAN/
├── base_model/              
├── dataset/                
├── function_fitting/        
├── image_classification/    
├── image_classification_conv/ 
├── solving_pde/            
```

## Quick Start: Running Experiments

We provide complete training and evaluation scripts for the four main tasks mentioned in the paper. All experiment logs and results will be saved in the `logs/` directory by default.

### Function Fitting

Run the following commands to perform one-dimensional or two-dimensional function fitting tasks.

- **Train a MetaKAN model for function fitting:**

```bash
  cd function_fitting

  python train_hyper.py \
      --model HyperKAN \
      --optimizer lbfgs \
      --lr 1 \
      --dataset I.6.20b \
      --layers_width 5 5 5 \
      --loss mse \
      --embedding_dim 1 \
      --hidden_dim 16 \

```
- **Train a KAN model for function fitting:**

```bash
  cd function_fitting

  python train.py \
      --model KAN \
      --optimizer lbfgs \
      --lr 1 \
      --dataset I.6.20b \
      --layers_width 5 5 5 \
      --loss mse \

```

### Image Classification (Fully Connected)

Perform image classification using a fully connected (MLP-style) structure on the MNIST or CIFAR-10 datasets.

- **Train MetaKAN on the MNIST dataset:**



```bash
  cd image_classfication
  
  python train_meta.py \
      --model MetaKAN \
      --optim_set double \
      --lr_h 1e-4 \
      --lr_e 1e-3 \
      --grid_size 5 \
      --spline_order 3 \
      --embedding_dim 1 \
      --hidden_dim 32 \
      --dataset MNIST \
      --batch_size 128 \
      --epochs 50 \

```

### Image Classification (Convolutional)

Perform image classification using a convolutional neural network with integrated MetaKAN layers.

- **Train a convolutional MetaKAN on the CIFAR-10 dataset:**



```bash
  cd image_classification_conv/
  
  python train_meta.py \
      --model MetaKAN8_M \
      --n_hypernets 1 \  
      --optim_set double \
      --lr_h 1e-4 \
      --lr_e 1e-3 \
      --grid_size 5 \
      --spline_order 3 \
      --embedding_dim 1 \
      --hidden_dim 32 \
      --dataset CIFAR10 \
      --batch_size 128 \
      --epochs 50 \
```
where n_hypernets means the number of  meta-learner

### Solving partial differential equations (PDEs)

Use MetaKAN in combination with physical information neural networks (PINNs) to solve partial differential equations.

- **Solving the one-dimensional Poisson's equation (1D Poisson's Equation):**

 

```bash
  cd solving_pde/
  
  python Poisson.py \
      --model HyperKAN \
      --dim 10 \
      --epochs 5000 \
      --embedding_dim 1 \
      --hidden_dim 32 \
```

  ```bash
      python AllenCahn.py \
      --model MetaKAN \
      --dim 10 \
      --epochs 5000 \
      --embedding_dim 1 \
      --hidden_dim 32 \
      --lr_h 1e-4 \
      --lr_e 1e-3 
  ```




If our work has been helpful to your research, please consider citing our paper:



```tex
@article{zhao2025improving,
  title={Improving Memory Efficiency for Training KANs via Meta Learning},
  author={Zhao, Zhangchi and Shu, Jun and Meng, Deyu and Xu, Zongben},
  journal={arXiv preprint arXiv:2506.07549},
  year={2025}
}
```

