from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
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
from model import GIN, GINTELL
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

SEEDS = 1

# def get_best_baseline_path(dataset_name):
#     l = glob.glob(f'results/{dataset_name}/*/results.json')
#     fl = [json.load(open(f)) for f in l]
#     df = pd.DataFrame(fl)
#     if df.shape[0] == 0: return None
#     df['fname'] = l
#     df = df.sort_values(by=['val_acc_mean', 'val_acc_std', 'test_acc_std'], ascending=[True,False,False])
#     df = df[df.fname.str.contains('nogumbel=False')]
#     fname = df.iloc[-1]['fname']
#     fname = fname.replace('/results.json', '')
#     return fname

def train_epoch(model_tell, loader, device, optimizer, num_classes, epoch, train_full, conv_reg=0.001, fc_reg=0.01, entropy_weight=1.0):
    model_tell.train()
    
    total_loss = [0]*(len(model_tell.convs)+1)
    total_correct = 0

    for batch_idx, data in enumerate(loader):
        try:
            if data.x is None:
                data.x = torch.ones((data.num_nodes, model_tell.num_features))
            if data.y.numel() == 0: continue
            if data.x.isnan().any(): continue
            if data.y.isnan().any(): continue
            y = data.y.reshape(-1).to(device).long()

            if train_full:
                for layer in model_tell.convs:
                    #layer.nn_0.phi_in.tau = 10
                    layer.nn_1.phi_in.tau = 10
                model_tell.fc.phi_in.tau = 10
                if model_tell.phi is not None:
                    model_tell.phi.tau = 10

            optimizer.zero_grad()
            loss = 0
            if data.edge_attr is None:
                out = model_tell(data.x.float().to(device), data.edge_index.to(device), None, data.batch.to(device))
            else:
                out = model_tell(data.x.float().to(device), data.edge_index.to(device), data.edge_attr.float().to(device), data.batch.to(device))
            pred = out.argmax(-1)
            loss += entropy_weight * (F.binary_cross_entropy(out.reshape(-1), torch.nn.functional.one_hot(y, num_classes=num_classes).float().reshape(-1)) + F.nll_loss(F.log_softmax(out, dim=-1), y.long()))
            #loss = F.binary_cross_entropy(out, y)
            if model_tell.phi is not None:
                #loss += conv_reg*(model_tell.phi.entropy + model_tell.phi.reg_loss)
                loss += 0
            for i,layer in enumerate(model_tell.convs):
                #print("Loss conv: ", layer.nn_0.reg_loss.item(), layer.nn_0.entropy_loss.item())
                layer_loss = conv_reg*(layer.nn_0.reg_loss + layer.nn_0.entropy_loss)
                layer_loss += conv_reg*(layer.nn_1.reg_loss + layer.nn_1.entropy_loss)
                total_loss[i] += layer_loss.item() / len(loader.dataset)
                #loss += layer_loss
                loss += 0

                # print("Loss conv 0: ", layer.nn_0.reg_loss.item(), layer.nn_0.entropy_loss.item())
                # print("Loss conv 1: ", layer.nn_1.reg_loss.item(), layer.nn_1.entropy_loss.item())

            #loss += fc_reg*(model_tell.fc.reg_loss + model_tell.fc.entropy_loss)
            loss += 0
            # print("Loss fc: ", model_tell.fc.reg_loss.item(), model_tell.fc.entropy_loss.item())

            loss.backward()
            zero_nan_gradients(model_tell, epoch, batch_idx)
            #torch.nn.utils.clip_grad_norm_(model_tell.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss[-1] += loss.item() * data.num_graphs / len(loader.dataset)
            total_correct += pred.eq(y).sum().item() / len(loader.dataset)
        except Exception as e:
            print(e)
            pass

    return total_loss, total_correct

@torch.no_grad()
def test_epoch(model, loader, device):
    model.eval()
    total_correct = 0
    for data in loader:
        if data.x is None:
            data.x = torch.ones((data.num_nodes, model.num_features))
        if data.y.numel() == 0: continue
        if data.x.isnan().any(): continue
        if data.y.isnan().any(): continue
        y = data.y.reshape(-1).to(device)
        if data.edge_attr is None:
            pred = model(data.x.float().to(device), data.edge_index.to(device), None, data.batch.to(device), tau=1000).argmax(-1)
        else:
            pred = model(data.x.float().to(device), data.edge_index.to(device), data.edge_attr.to(device), data.batch.to(device), tau=1000).argmax(-1)
        total_correct += pred.eq(y).sum().item()
    val_acc = total_correct / len(loader.dataset)
    
    return val_acc

def train_seed(dataset_name, args, seed, device):
    set_seed(seed)

        
    path = create_folder_logic_no_teacher(dataset_name, args, seed=seed)
    shutil.rmtree(path)
    path = create_folder_logic_no_teacher(dataset_name, args, seed=seed)

    os.mkdir(os.path.join(path, 'code'))
    for f in glob.glob('*.py'):
        shutil.copy(f, os.path.join(path, 'code'))

    with open(os.path.join(path, 'args.json'), 'w') as f:
        args = {k: (v.item() if hasattr(v, 'item') else v) for k,v in args.items()}
        json.dump(args, f)


    dataset = get_dataset(dataset_name)

    print(f'Training logic model on {dataset_name}')

    
    num_classes = dataset.num_classes
    num_features = dataset.num_features
    num_features_edge = dataset.num_edge_features
    print("Num features:", num_features)
    print("Num edge features:", num_features_edge)

    if num_features == 0: num_features = 10
    
    indices = list(range(len(dataset)))
    train_indices, val_test_indices = train_test_split(indices, test_size=0.2, shuffle=True, stratify=dataset.data.y, random_state=seed)

    val_indices = val_test_indices[:len(val_test_indices)//2]
    test_indices = val_test_indices[len(val_test_indices)//2:]

    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]

    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    with open(os.path.join(path, 'data.pkl'), 'wb') as f:
        pickle.dump({
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
        }, f)

    if dataset_name == 'PROTEINS':
        model_tell = GINTELL(num_features=num_features, num_features_edge=num_features_edge, num_classes=num_classes, hidden_dim=args['hidden_dim'], num_layers=args['num_layers'], 
                             dropout=args['dropout'], negative_concatenate=args['negative_concatenate'], edge_again=args['edge_again'], input_binary=False).to(device)
    else:
        model_tell = GINTELL(num_features=num_features, num_features_edge=num_features_edge, num_classes=num_classes, hidden_dim=args['hidden_dim'], num_layers=args['num_layers'],
                             dropout=args['dropout'], negative_concatenate=args['negative_concatenate'], edge_again=args['edge_again']).to(device)
    
    optimizer = torch.optim.AdamW(model_tell.parameters(), lr=args['lr'], weight_decay=args['l2'])

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=100, min_lr=1e-5, verbose=True)
    
    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    train_accs = []
    val_accs = []
    test_accs = []
    patience = 0
    max_patience = 1000
    
    for epoch in range(args['epochs']):
        train_loss, train_acc = train_epoch(model_tell, train_loader, device, optimizer, num_classes, epoch, train_full = epoch>args['warmup_epochs'], 
                                conv_reg=args['conv_reg'], fc_reg=args['fc_reg'], entropy_weight=args['entropy_weight'])
        val_acc = test_epoch(model_tell, val_loader, device)
        test_acc = test_epoch(model_tell, test_loader, device)

        if epoch > args['warmup_epochs']:
            scheduler.step(val_acc)
            patience += 1
        
        if epoch > args['warmup_epochs'] and val_acc >= best_val_acc:
            torch.save(model_tell, os.path.join(path, 'best.pt'))
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience = 0
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Train Acc: {train_acc}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
            print(f'\t\t Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}')
            
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        if patience >= max_patience:
            break

        
    
    torch.save(model_tell, os.path.join(path, 'last.pt'))
    # model_tell.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    model_tell = torch.load(os.path.join(path, 'best.pt'))

    val_acc = test_epoch(model_tell, val_loader, device)
    test_acc = test_epoch(model_tell, test_loader, device)

    results = {
        'seed': seed,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }

    return results

def eval_seed(dataset_name, args, seed, device):
    set_seed(seed)

    path = create_folder_logic_no_teacher(dataset_name, args, seed=seed)

    dataset = get_dataset(dataset_name)


    num_classes = dataset.num_classes
    num_features = dataset.num_features
    num_features_edge = dataset.num_edge_features

    if num_features == 0: num_features = 10
    
    data = pickle.load(open(os.path.join(path, 'data.pkl'), 'rb'))

    train_dataset = dataset[data['train_indices']]
    val_dataset = dataset[data['val_indices']]
    test_dataset = dataset[data['test_indices']]
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    if dataset_name == 'PROTEINS':
        model_tell = GINTELL(num_features=num_features, num_features_edge=num_features_edge, num_classes=num_classes, hidden_dim=args['hidden_dim'], num_layers=args['num_layers'], 
                             dropout=args['dropout'], negative_concatenate=args['negative_concatenate'], edge_again=args['edge_again'], input_binary=False).to(device)
    else:
        model_tell = GINTELL(num_features=num_features, num_features_edge=num_features_edge, num_classes=num_classes, hidden_dim=args['hidden_dim'], num_layers=args['num_layers'],
                             dropout=args['dropout'], negative_concatenate=args['negative_concatenate'], edge_again=args['edge_again']).to(device)
    model_tell = torch.load(os.path.join(path, 'best.pt'))
    # model_tell.load_state_dict(torch.load('best.pt'))

    val_acc = test_epoch(model_tell, val_loader, device)
    test_acc = test_epoch(model_tell, test_loader, device)

    results = {
        'seed': seed,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }

    return results


def train_eval(dataset_name, args):
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    print('Device:', device)
    print('Args:', args)
    #baseline_args = json.load(open(os.path.join(baseline_path, '0', 'args.json'), 'r'))
    
    seed_todo = args.pop('seed', None)
    only_eval = args.pop('only_eval', False)
    
    path = create_folder_logic_no_teacher(dataset_name, args)

    seeds = range(SEEDS)
    if seed_todo is not None:
        seeds = [seed_todo]
    
    results = []
    if not only_eval:
        for seed in seeds:
            results.append(train_seed(dataset_name, args, seed, device))

    print(results)

    if only_eval or seed_todo is not None:
        results = []
        for seed in range(SEEDS):
            try:
                r = eval_seed(dataset_name, args, seed, device)
                results.append(r)
                print(r)
            except Exception as e: print(e)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(path, 'total_results.csv'))

    ret = {
        'val_acc_mean': df['val_acc'].mean(),
        'test_acc_mean': df['test_acc'].mean(),
        'val_acc_std': df['val_acc'].std(),
        'test_acc_std': df['test_acc'].std()
    }

    with open(os.path.join(path, 'results.json'), 'w') as f:
        json.dump(ret, f)

    return ret
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train_baseline.py')

    parser.add_argument('--dataset',       default='NCI1', type=str,     help='Dataset to use')
    #parser.add_argument('--baseline_path', default=None,       type=str,     help='Baseline path')
    parser.add_argument('--epochs',        default=5000,       type=int,     help='Epochs')
    parser.add_argument('--warmup_epochs', default=3000,       type=int,     help='Epochs')
    parser.add_argument('--batch_size',    default=32,         type=int,     help='Batch Size')
    parser.add_argument('--lr',            default=0.01,      type=float,   help='Learning Rate')
    parser.add_argument('--l2',            default=0.0,      type=float,     help='Weight decay')
    parser.add_argument('--conv_reg',      default=0.0,      type=float,   help='Conv layer regularization')
    parser.add_argument('--fc_reg',        default=0.0,      type=float,    help='Last layer regularization')
    parser.add_argument('--dropout',       default=0.0,        type=float,   help='Dropout')
    parser.add_argument('--hidden_dim',    default=128,         type=int,     help='Hidden Dimension')
    parser.add_argument('--num_layers',    default=5,          type=int,     help='Number of Convolutional Layers')
    parser.add_argument('--negative_concatenate', default=2,       type=int,     help='Use negative concatenation')
    parser.add_argument('--edge_again',   action='store_true',              help='Use edge features again in MLP after aggregation')
    parser.add_argument('--entropy_weight', default=1.0,      type=float,    help='Weight of entropy loss')
    parser.add_argument('--only_eval',    action='store_true',              help='Only evaluate the model')
    parser.add_argument('--seed',          default=None,      type=int,    help='Random seed')
    #Poi rimuovere
    parser.add_argument('--activation',     default='free_weight', type=str,   help='Activation function for weights')

    args = parser.parse_args().__dict__
    
    dataset_name = args.pop('dataset')
    # baseline_path = args.pop('baseline_path')
    # if baseline_path is None:
    #     baseline_path = get_best_baseline_path(dataset_name)
    #     print('Baseline path found:', baseline_path)
    train_eval(dataset_name, args)

    