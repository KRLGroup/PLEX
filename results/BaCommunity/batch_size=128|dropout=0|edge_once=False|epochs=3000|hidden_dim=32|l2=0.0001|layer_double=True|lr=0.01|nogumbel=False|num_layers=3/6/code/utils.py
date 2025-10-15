import torch
import random
import numpy as np
import os
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset, Planetoid, OGB_MAG, MoleculeNet
from ba_multi_shapes import BAMultiShapesDataset
from syn_dataset import SynGraphDataset
from spmotif_dataset import *
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
from utils import *
from sklearn.model_selection import train_test_split
import shutil
import glob
import pandas as pd
import argparse
import pickle
import json

def create_folder(dataset_name, args, seed=None):
    args_s = '|'.join([f"{k}={args[k]}" for k in sorted(args.keys())])
    path = f'results/{dataset_name}/{args_s}'    
    if seed is not None:
        path = f"{path}/{seed}"
    os.makedirs(path, exist_ok=True)
    return path

def create_folder_logic(dataset_name, args, baseline_args, seed=None):
    args_s = '|'.join([f"{k}={args[k]}" for k in sorted(args.keys())])
    baseline_args_s = '|'.join([f"{k}={baseline_args[k]}" for k in sorted(baseline_args.keys())])
    path = f'results_logic/{dataset_name}/{args_s}/{baseline_args_s}'    
    if seed is not None:
        path = f"{path}/{seed}"
    os.makedirs(path, exist_ok=True)
    return path

def create_folder_logic_no_teacher(dataset_name, args, seed=None):
    args_s = '|'.join([f"{k}={args[k]}" for k in sorted(args.keys())])
    path = f'results_logic_no_teacher/{dataset_name}/{args_s}'    
    if seed is not None:
        path = f"{path}/{seed}"
    os.makedirs(path, exist_ok=True)
    return path

def get_dataset(dataset_name):
    if dataset_name == 'Ba2Motifs':
        return  SynGraphDataset(root='data/ba_2motifs', name='ba_2motifs')
    elif dataset_name == 'Ba2MotifsNoisy':
        return  SynGraphDataset(root='data/ba_2motifs', name='ba_2motifsnoisy')
    elif dataset_name == 'TreeGrid':
        return  SynGraphDataset(root='data/tree_grid', name='tree_grid')
    elif dataset_name == 'BaShapes':
        return  SynGraphDataset(root='data/ba_shapes', name='ba_shapes')
    elif dataset_name == 'BaCommunity':
        return  SynGraphDataset(root='data/ba_community', name='ba_community')

    elif dataset_name == 'SPMotif':
        return SPMotif(root='data/SPMotif-0.333', mode='train', transform=None)

    elif dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        return Planetoid(root=f'data/{dataset_name}', name=dataset_name, split = 'full')
    elif dataset_name == 'OGB_MAG':
        return OGB_MAG(root=f'data/{dataset_name}')
    elif dataset_name == 'BBBP':
        return MoleculeNet(name=dataset_name, root=f'data/{dataset_name}')
    elif dataset_name == 'BaMultiShapes':
        return BAMultiShapesDataset(root=f'data/{dataset_name}')
    return TUDataset(root=f'data/{dataset_name}', name=dataset_name, use_node_attr=True, use_edge_attr=True)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return torch.Generator().manual_seed(seed)

# def zero_nan_gradients(model):
#     for param in model.parameters():
#         if param.grad is not None:
#             param.grad[param.grad != param.grad] = 0  # Set NaN gradients to 0

def zero_nan_gradients(model, epoch=None, batch_idx=None):
    for name, param in model.named_parameters():  # uso named_parameters per sapere quale layer
        if param.grad is not None:
            # if epoch % 100 == 0 and batch_idx == 0:  # Log ogni 100 epoche, solo per il primo batch
            #     # --- Statistiche gradienti ---
            # grad_mean = param.grad.mean().item()
            # grad_min = param.grad.min().item()
            # grad_max = param.grad.max().item()
            # print(f"[Grad] {name} | mean: {grad_mean:.6f}, min: {grad_min:.6f}, max: {grad_max:.6f}")

                # --- Statistiche pesi ---
                # weight_mean = param.data.mean().item()
                # weight_min = param.data.min().item()
                # weight_max= param.data.max().item()
                # print(f"[Peso] {name} | mean: {weight_mean:.6f}, min: {weight_min:.6f}, max: {weight_max:.6f}")
            # --- Controllo NaN nei gradienti ---
            nan_mask = param.grad != param.grad  # True dove ci sono NaN
            if nan_mask.any():
                print(f"⚠️ NaN trovato nei gradienti di: {name}, numero di NaN: {nan_mask.sum().item()}")
                # grad_mean = param.grad.mean().item()
                # grad_min = param.grad.min().item()
                # grad_max = param.grad.max().item()
                # print(f"[Grad] {name} | mean: {grad_mean:.6f}, min: {grad_min:.6f}, max: {grad_max:.6f}")
                param.grad[nan_mask] = 0  # Sostituisce i NaN con 0
                # weight_mean = param.data.mean().item()
                # weight_min = param.data.min().item()
                # weight_max= param.data.max().item()
                # print(f"[Peso] {name} | mean: {weight_mean:.6f}, min: {weight_min:.6f}, max: {weight_max:.6f}")
