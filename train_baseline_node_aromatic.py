import torch
import torch.nn.functional as F
from spmotif_dataset import *
from utils import *
import shutil
import glob
import pandas as pd
import argparse
import pickle
import json
from model_node import GIN
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import time


SEEDS = 5

def train_epoch(model, loader, device, optimizer, num_classes):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if batch.edge_attr is None:
            out = model(batch.x.float(), batch.edge_index, None, tau=1)
        else:
            out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), tau=1)

        # out è di forma [num_nodes_in_batch, num_classes]
        # batch.y è di forma [num_nodes_in_batch]
        pred = out.argmax(-1)

        loss = F.binary_cross_entropy(
            out.reshape(-1), 
            torch.nn.functional.one_hot(batch.y, num_classes=num_classes).float().reshape(-1)
        )
        nll_loss = F.nll_loss(F.log_softmax(out, dim=-1), batch.y.long())
        loss = nll_loss + loss
        
        loss.backward()
        zero_nan_gradients(model)
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        total_correct += pred.eq(batch.y).sum().item()
        total_samples += batch.y.size(0)

    train_loss = total_loss / len(loader.dataset)
    train_acc = total_correct / total_samples

    return train_loss, train_acc

@torch.no_grad()
def test_epoch(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        if batch.edge_attr is None:
            out = model(batch.x.float(), batch.edge_index, None, tau=1000)
        else:
            out = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), tau=1000)
        
        pred = out.argmax(-1)
        total_correct += pred.eq(batch.y).sum().item()
        total_samples += batch.y.size(0)
    
    acc = total_correct / total_samples

    return acc

def train_seed(dataset_name, args, seed, device):
    set_seed(seed)

    path = create_folder(dataset_name, args, seed=seed)
    shutil.rmtree(path)
    path = create_folder(dataset_name, args, seed=seed)

    os.mkdir(os.path.join(path, 'code'))
    for f in glob.glob('*.py'):
        shutil.copy(f, os.path.join(path, 'code'))

    with open(os.path.join(path, 'args.json'), 'w') as f:
        args_copy = {k: (v.item() if hasattr(v, 'item') else v) for k,v in args.items()}
        json.dump(args_copy, f)

    dataset = get_dataset(dataset_name)
    
    num_classes = dataset.num_classes
    num_features = dataset.num_features
    num_features_edge = dataset.num_edge_features

    print(f'Num classes: {num_classes}, Num features: {num_features}, Num edge features: {num_features_edge}')
    
    # Split dataset into train/val/test
    from sklearn.model_selection import train_test_split
    
    indices = list(range(len(dataset)))
    
    # Collect all labels for stratification
    all_labels = []
    for idx in indices:
        data = dataset[idx]
        # Use majority label or first label for stratification
        all_labels.append(data.y[0].item())
    
    train_indices, temp_indices = train_test_split(
        indices, test_size=0.2, shuffle=True, 
        stratify=all_labels, random_state=seed
    )
    
    temp_labels = [all_labels[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, shuffle=True,
        stratify=temp_labels, random_state=seed
    )
    
    # Create subsets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)
    
    with open(os.path.join(path, 'data.pkl'), 'wb') as f:
        pickle.dump({
            'dataset': dataset,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        }, f)

    model = GIN(
        num_features=num_features, 
        num_features_edge=num_features_edge, 
        num_classes=num_classes, 
        hidden_dim=args['hidden_dim'], 
        num_layers=args['num_layers'], 
        nogumbel=args['nogumbel'], 
        edge_once=args['edge_once'], 
        layer_double=args['layer_double']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['l2'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=100, min_lr=1e-5, verbose=True)

    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    train_accs = []
    val_accs = []
    test_accs = []

    
    for epoch in range(args['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, device, optimizer, num_classes)
        val_acc = test_epoch(model, val_loader, device)
        test_acc = test_epoch(model, test_loader, device)
        scheduler.step(val_acc)

        if val_acc >= best_val_acc:

            torch.save(model.state_dict(), os.path.join(path, 'best.pt'))
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(f'\t\t Best Val Acc: {best_val_acc:.4f}, Best Test Acc: {best_test_acc:.4f}')

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
    
    torch.save(model.state_dict(), os.path.join(path, 'last.pt'))
    model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))

    val_acc = test_epoch(model, val_loader, device)
    test_acc = test_epoch(model, test_loader, device)

    results = {
        'seed': seed,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }

    return results


def eval_seed(dataset_name, args, seed, device):
    set_seed(seed)

    path = create_folder(dataset_name, args, seed=seed)

    dataset = get_dataset(dataset_name)

    num_classes = dataset.num_classes
    num_features = dataset.num_features
    num_features_edge = dataset.num_edge_features

    print(f'Num classes: {num_classes}, Num features: {num_features}, Num edge features: {num_features_edge}')

    # Load indices from training
    with open(os.path.join(path, 'data.pkl'), 'rb') as f:
        data_dict = pickle.load(f)
        val_indices = data_dict['val_indices']
        test_indices = data_dict['test_indices']
    
    from torch.utils.data import Subset
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)

    model = GIN(
        num_features=num_features, 
        num_features_edge=num_features_edge, 
        num_classes=num_classes, 
        hidden_dim=args['hidden_dim'], 
        num_layers=args['num_layers'], 
        nogumbel=args['nogumbel'], 
        edge_once=args['edge_once'], 
        layer_double=args['layer_double']
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))

    val_acc = test_epoch(model, val_loader, device)
    test_acc = test_epoch(model, test_loader, device)

    results = {
        'seed': seed,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }

    return results

def train(dataset_name, args):
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    path = create_folder(dataset_name, args)

    results = []
    for seed in range(10):
        results.append(train_seed(dataset_name, args, seed, device))

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

def train_eval(dataset_name, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    seed_todo = args.pop('seed', None)
    only_eval = args.pop('only_eval', False)

    path = create_folder(dataset_name, args)
    print(path)

    seeds = range(SEEDS)
    if seed_todo is not None:
        seeds = [seed_todo]

    results = []
    if not only_eval:
        for seed in seeds:
            start_time = time.time()
            result = train_seed(dataset_name, args, seed, device)
            result['training_time'] = (time.time() - start_time) / 60
            results.append(result)

    print(results)

    if only_eval or seed_todo is not None:
        results = []
        for seed in range(SEEDS):
            r = eval_seed(dataset_name, args, seed, device)
            print(r)
            results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(path, 'total_results.csv'))

    ret = {
        'val_acc_mean': df['val_acc'].mean(),
        'test_acc_mean': df['test_acc'].mean(),
        'val_acc_std': df['val_acc'].std(),
        'test_acc_std': df['test_acc'].std(),
    }

    with open(os.path.join(path, 'results.json'), 'w') as f:
        json.dump(ret, f)

    return ret
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train_baseline.py')

    parser.add_argument('--dataset',       default='AromaticCarbon', type=str,     help='Dataset to use')
    parser.add_argument('--epochs',        default=100,       type=int,     help='Epochs')
    parser.add_argument('--hidden_dim',    default=32,        type=int,     help='Hid Dim')
    parser.add_argument('--batch_size',    default=32,         type=int,     help='Batch Size')
    parser.add_argument('--num_layers',    default=5,          type=int,     help='Number of Convolutional Layers')
    parser.add_argument('--dropout',       default=0.0,       type=float,   help='Dropout')
    parser.add_argument('--lr',            default=0.001,      type=float,   help='Learning Rate')
    parser.add_argument('--l2',            default=1e-5,      type=float,   help='Weight Decay')
    parser.add_argument('--nogumbel',      action='store_true',             help='Use ReLU instead of Gumbel-Sigmoid')
    parser.add_argument('--edge_once',     action='store_true',             help='Use edge features only in first layer')
    parser.add_argument('--layer_double',  action='store_true',             help='Use double layers')
    parser.add_argument('--seed',          default=None,      type=int,     help='Set a specific seed to run')
    parser.add_argument('--only_eval',    action='store_true',             help='Only evaluate the model')

    args = parser.parse_args().__dict__
    
    dataset_name = args.pop('dataset')
    train_eval(dataset_name, args)