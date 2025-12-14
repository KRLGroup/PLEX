import torch
import torch.nn.functional as F
from spmotif_dataset import *
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from utils import *
import shutil
import glob
import pandas as pd
import argparse
import pickle
import json
from model_link import GIN
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score

SEEDS = 1

def train_epoch(model, loader, device, optimizer):
    model.train()
    total_loss = 0
    total_auc = 0
    count = 0

    for data in loader:
        data = data.to(device)


        optimizer.zero_grad()

        if data.edge_attr is None:
            out = model.forward(data.x.float(), data.edge_index, None, data.edge_label_index)
        else:
            out = model.forward(data.x.float(), data.edge_index, data.edge_attr.float(), data.edge_label_index)


        loss = F.binary_cross_entropy(out.view(-1), data.edge_label.float())
                
        loss.backward()
        zero_nan_gradients(model)
        optimizer.step()
        auc = roc_auc_score(data.edge_label.float().cpu().numpy(), out.detach().cpu().numpy())

        total_loss += loss.item()
        total_auc += auc
        count += 1

    return total_loss / count, total_auc / count


@torch.no_grad()
def test_epoch(model, data, device):
    model.eval()
    data = data.to(device)
    
    if data.edge_attr is None:
        out = model.forward(data.x.float(), data.edge_index, None, data.edge_label_index)
    else:
        out = model.forward(data.x.float(), data.edge_index, data.edge_attr.float(), data.edge_label_index)
    
    auc = roc_auc_score(data.edge_label.float().cpu().numpy(), out.cpu().numpy())
    return auc
    

def train_seed(dataset_name, args, seed, device):
    set_seed(seed)

    path = create_folder(dataset_name, args, seed=seed)
    shutil.rmtree(path)
    path = create_folder(dataset_name, args, seed=seed)

    os.mkdir(os.path.join(path, 'code'))
    for f in glob.glob('*.py'):
        shutil.copy(f, os.path.join(path, 'code'))

    with open(os.path.join(path, 'args.json'), 'w') as f:
        args = {k: (v.item() if hasattr(v, 'item') else v) for k,v in args.items()}
        json.dump(args, f)

    dataset = get_dataset(dataset_name)
    data = dataset[0]

    print(dataset)
    print(data)
    
    num_features = dataset.num_features
    num_features_edge = dataset.num_edge_features

    print(f'Num features: {num_features}, Num edge features: {num_features_edge}')


    if num_features == 0: num_features = 10

    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.10,
        neg_sampling_ratio=1.0,  # Ratio 1:1 positive-negative
        add_negative_train_samples=False,  # Aggiungi negative samples al training
        is_undirected=True,  # IMPORTANTE: grafo non diretto
    )
    train_data, val_data, test_data = transform(data)
    print(f"Train labels uniche: {train_data.edge_label.unique()}")
    print(f"Train label counts: {torch.bincount(train_data.edge_label.long())}")
    print(f"\nTrain data: {train_data}")
    print(f"  - edge_index: {train_data.edge_index.size()}")
    print(f"  - edge_label_index: {train_data.edge_label_index.size()}")
    print(f"  - edge_label: {train_data.edge_label.size()}")
    print(f"  - Positive edges: {train_data.edge_label.sum().item()}")
    print(f"  - Negative edges: {(train_data.edge_label == 0).sum().item()}")
    
    print(f"\nVal data: {val_data}")
    print(f"  - edge_index: {val_data.edge_index.size()}")
    print(f"  - edge_label_index: {val_data.edge_label_index.size()}")
    print(f"  - edge_label: {val_data.edge_label.size()}")
    print(f"  - Positive edges: {val_data.edge_label.sum().item()}")
    print(f"  - Negative edges: {(val_data.edge_label == 0).sum().item()}")
    
    print(f"\nTest data: {test_data}")
    print(f"  - edge_index: {test_data.edge_index.size()}")
    print(f"  - edge_label_index: {test_data.edge_label_index.size()}")
    print(f"  - edge_label: {test_data.edge_label.size()}")
    print(f"  - Positive edges: {test_data.edge_label.sum().item()}")
    print(f"  - Negative edges: {(test_data.edge_label == 0).sum().item()}")

    # Save splits
    with open(os.path.join(path, 'data.pkl'), 'wb') as f:
        pickle.dump({
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
        }, f)


    
    model = GIN(num_features=num_features, num_features_edge=num_features_edge, hidden_dim=args['hidden_dim'], num_layers=args['num_layers'], nogumbel=args['nogumbel'],
                edge_once = args['edge_once'], layer_double=args['layer_double']).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['l2'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=100, min_lr=1e-5, verbose=True)

    # Training loop
    best_val_auc = 0
    best_test_auc = 0
    train_aucs = []
    val_aucs = []
    test_aucs = []
    max_patience = 500
    patience = 0

    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=[10] * args['num_layers'],
        batch_size=128,
        neg_sampling_ratio=2.0,
        shuffle=True,
    )

    for epoch in range(args['epochs']):

        train_loss, train_auc = train_epoch(model, train_loader, device, optimizer)
        val_auc = test_epoch(model, val_data, device)
        test_auc = test_epoch(model, test_data, device)
        scheduler.step(val_auc)

        if val_auc >= best_val_auc:
            patience = 0
            torch.save(model.state_dict(), os.path.join(path, 'best.pt'))
            best_val_auc = val_auc
            best_test_auc = test_auc
        elif epoch > args['epochs']//2: patience += 1
        
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')
        print(f'\t\t Best Val AUC: {best_val_auc:.4f}, Best Test AUC: {best_test_auc:.4f}')

        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        test_aucs.append(test_auc)

        if patience >= max_patience: break
    
    torch.save(model.state_dict(), os.path.join(path, 'last.pt'))
    model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))

    val_auc = test_epoch(model, val_data, device)
    test_auc = test_epoch(model, test_data, device)

    results = {
        'seed': seed,
        'val_auc': val_auc,
        'test_auc': test_auc,
    }

    return results
    
def train(dataset_name, args):
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

    path = create_folder(dataset_name, args)

    results = []
    for seed in range(5):
        results.append(train_seed(dataset_name, args, seed, device))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(path, 'total_results.csv'))

    ret = {
        'val_auc_mean': df['val_auc'].mean(),
        'test_auc_mean': df['test_auc'].mean(),
        'val_auc_std': df['val_auc'].std(),
        'test_auc_std': df['test_auc'].std()
    }

    with open(os.path.join(path, 'results.json'), 'w') as f:
        json.dump(ret, f)

    return ret
    
def eval_seed(dataset_name, args, seed, device):
    set_seed(seed)

    path = create_folder(dataset_name, args, seed=seed)
    
    dataset = get_dataset(dataset_name)
    data = dataset[0]

    
    num_features = dataset.num_features
    num_features_edge = dataset.num_edge_features

    if num_features == 0: num_features = 10
    
    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.10,
        neg_sampling_ratio=1.0,  # Ratio 1:1 positive-negative
        add_negative_train_samples=False,  # Aggiungi negative samples al training
        is_undirected=True,  # IMPORTANTE: grafo non diretto
    )

    train_data, val_data, test_data = transform(data)

    model = GIN(num_features=num_features, num_features_edge=num_features_edge, hidden_dim=args['hidden_dim'], num_layers=args['num_layers'], nogumbel=args['nogumbel'],
                edge_once=args['edge_once'], layer_double=args['layer_double']).to(device)

    model.load_state_dict(torch.load(os.path.join(path, 'best.pt')))

    val_auc = test_epoch(model, val_data, device)
    test_auc = test_epoch(model, test_data, device)

    results = {
        'seed': seed,
        'val_auc': val_auc,
        'test_auc': test_auc,
    }

    return results


def train_eval(dataset_name,  args):
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
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
        'val_auc_mean': df['val_auc'].mean(),
        'test_auc_mean': df['test_auc'].mean(),
        'val_auc_std': df['val_auc'].std(),
        'test_auc_std': df['test_auc'].std()
    }

    with open(os.path.join(path, 'results.json'), 'w') as f:
        json.dump(ret, f)

    return ret
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train_baseline.py')

    parser.add_argument('--dataset',       default='Cora', type=str,     help='Dataset to use')
    parser.add_argument('--epochs',        default=3000,       type=int,     help='Epochs')
    parser.add_argument('--hidden_dim',    default=32,        type=int,     help='Hid Dim')
    parser.add_argument('--num_layers',    default=5,          type=int,     help='Number of Convolutional Layers')
    parser.add_argument('--dropout',       default=0.5,       type=float,   help='Dropout')
    parser.add_argument('--lr',            default=0.001,      type=float,   help='Learning Rate')
    parser.add_argument('--l2',            default=0.1,      type=float,   help='Weight Decay')
    parser.add_argument('--nogumbel',      action='store_true',             help='Use ReLU instead of Gumbel-Sigmoid')
    parser.add_argument('--edge_once',    action='store_true',              help='Use edge features only in first layer')
    parser.add_argument('--layer_double',  action='store_true',              help='Use double layers')
    parser.add_argument('--only_eval',    action='store_true',              help='Only evaluate the model')
    parser.add_argument('--seed',          default=None,      type=int,    help='Random seed')

    args = parser.parse_args().__dict__
    
    dataset_name = args.pop('dataset')
    train_eval(dataset_name, args)

    
