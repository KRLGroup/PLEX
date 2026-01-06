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
from model_node import GIN, GINTELL
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

SEEDS = 5


def get_best_baseline_path(dataset_name):
    l = glob.glob(f'results/{dataset_name}/*/results.json')
    fl = [json.load(open(f)) for f in l]
    df = pd.DataFrame(fl)
    if df.shape[0] == 0: return None
    df['fname'] = l
    df = df.sort_values(by=['val_acc_mean', 'val_acc_std', 'test_acc_std'], ascending=[True,False,False])
    df = df[df.fname.str.contains('nogumbel=False')]
    fname = df.iloc[-1]['fname']
    fname = fname.replace('/results.json', '')
    return fname

def train_epoch(model, model_tell, loader, device, optimizer, num_classes, train_full=True, conv_reg=0.001, fc_reg=0.01, only_teacher=0):
    model_tell.train()
    
    total_loss = [0]*(len(model_tell.convs)+1)
    total_correct = [0]*(len(model_tell.convs)+1)
    total_samples = 0

    if train_full:
        if model_tell.phi_node is not None:
            model_tell.phi_node.tau = 10
        if model_tell.phi_edge is not None:
            model_tell.phi_edge.tau = 10
        for layer in model_tell.convs:
            layer.nn_1.phi_in.tau = 10

    for batch in loader:
        batch = batch.to(device)
        y = batch.y
        
        optimizer.zero_grad()
        
        # Get teacher outputs
        with torch.no_grad():
            if batch.edge_attr is None:
                layers_x, layers_y, layers_y_interm = model.forward_e(batch.x.float(), batch.edge_index, None, tau=1000)
            else:
                layers_x, layers_y, layers_y_interm = model.forward_e(batch.x.float(), batch.edge_index, batch.edge_attr.float(), tau=1000)

        loss = 0
        last_layer_out = None

        if train_full == False or (train_full == True and (only_teacher == 1 or only_teacher == 2)):

            if batch.edge_attr is not None and model_tell.negative_concatenate!=0:
                if model_tell.phi_edge is not None:
                    edge_attr = batch.edge_attr.float()
                    edge_attr = model_tell.phi_edge(edge_attr)
                    edge_attr = torch.hstack([edge_attr, 1-edge_attr]).float()
                else:
                    edge_attr = torch.hstack([batch.edge_attr, 1-batch.edge_attr]).float()

            elif batch.edge_attr is not None and model_tell.negative_concatenate==0:
                if model_tell.phi_edge is not None:
                    edge_attr = batch.edge_attr.float()
                    edge_attr = model_tell.phi_edge(edge_attr).float()
                else:
                    edge_attr = batch.edge_attr.float()

            for i, (layer, layer_x, layer_y, layer_y_interm) in enumerate(zip(model_tell.convs, layers_x[:-1], layers_y[:-1], layers_y_interm)):
                
                if i==0 and not model_tell.input_binary:
                    layer_x = model_tell.phi_node(layer_x)
                    loss_binarization_input = conv_reg*(model_tell.phi_node.entropy + model_tell.phi_node.reg_loss)
                    loss += loss_binarization_input

                if i==0 and not model_tell.edge_binary:
                    loss_binarization_edge = conv_reg*(model_tell.phi_edge.entropy + model_tell.phi_edge.reg_loss)
                    loss += loss_binarization_edge
                
                if (i==0 and model_tell.negative_concatenate==1) or model_tell.negative_concatenate==2:
                    layer_x = torch.hstack([layer_x, 1-layer_x])

                if i == 0 and batch.edge_attr is not None:
                    layer_out, layer_out_interm = layer(layer_x, batch.edge_index, edge_attr)
                elif i!=0 and batch.edge_attr is not None and model_tell.edge_again:
                    layer_out, layer_out_interm = layer(layer_x, batch.edge_index, edge_attr)
                else:
                    layer_out, layer_out_interm = layer(layer_x, batch.edge_index, None)
                
                layer_loss = F.binary_cross_entropy(layer_out.reshape(-1), layer_y.reshape(-1)) + conv_reg* (layer.nn_1.reg_loss + layer.nn_1.entropy_loss)
                layer_loss += F.binary_cross_entropy(layer_out_interm.reshape(-1), layer_y_interm.reshape(-1)) + conv_reg* (layer.nn_0.reg_loss + layer.nn_0.entropy_loss)
                loss += layer_loss

                if train_full and i!=0:
                    layer_x = last_layer_out

                    if model_tell.negative_concatenate == 2:
                        layer_x = torch.hstack([layer_x, 1-layer_x])
                    
                    if batch.edge_attr is not None and model_tell.edge_again:
                        layer_out, layer_out_interm = layer(layer_x, batch.edge_index, edge_attr)
                    else:
                        layer_out, layer_out_interm = layer(layer_x, batch.edge_index, None)
                    
                    layer_loss = F.binary_cross_entropy(layer_out.reshape(-1), layer_y.reshape(-1)) + conv_reg* (layer.nn_1.reg_loss + layer.nn_1.entropy_loss)
                    layer_loss += F.binary_cross_entropy(layer_out_interm.reshape(-1), layer_y_interm.reshape(-1)) + conv_reg* (layer.nn_0.reg_loss + layer.nn_0.entropy_loss)
                    loss += layer_loss

                last_layer_out = layer_out
                total_loss[i] += layer_loss.item() * batch.num_graphs
                total_correct[i] += ((layer_out >= 0.5).long() == layer_y).sum().item() / layer_y.shape[-1]

            layer_x = layers_x[-1]
            if model_tell.negative_concatenate==2:
                layer_x = torch.hstack([layer_x, 1-layer_x])
            out = model_tell.fc(layer_x)
            pred = out.argmax(-1)

            if only_teacher == 2 or only_teacher == 0:
                layer_y = layers_y[-1]
                layer_y = (layer_y > 0.5).float()
                layer_y_true = layer_y.argmax(-1)
                loss_fc_teacher = F.binary_cross_entropy(out.reshape(-1), layer_y.reshape(-1)) + F.nll_loss(F.log_softmax(out, dim=-1), layer_y_true)
            else:
                loss_fc_teacher = F.binary_cross_entropy(out.reshape(-1), torch.nn.functional.one_hot(y, num_classes=num_classes).float().reshape(-1)) + F.nll_loss(F.log_softmax(out, dim=-1), y.long())
            loss += loss_fc_teacher

        if train_full:
            if batch.edge_attr is not None:
                out = model_tell(batch.x.float(), batch.edge_index, batch.edge_attr.float())
            else:
                out = model_tell(batch.x.float(), batch.edge_index, None)
            
            pred = out.argmax(-1)

            loss_fc_teacher_full = F.binary_cross_entropy(out.reshape(-1), torch.nn.functional.one_hot(y, num_classes=num_classes).float().reshape(-1)) + F.nll_loss(F.log_softmax(out, dim=-1), y.long())
            loss += loss_fc_teacher_full

        loss_fc_reg = fc_reg*(model_tell.fc.reg_loss + model_tell.fc.entropy_loss)
        loss += loss_fc_reg
        
        loss.backward()
        zero_nan_gradients(model_tell)
        optimizer.step()
        
        total_loss[-1] += loss.item() * batch.num_graphs
        total_correct[-1] += pred.eq(y).sum().item()
        total_samples += y.size(0)

    # Average over all samples
    total_loss = [l / len(loader.dataset) for l in total_loss]
    total_correct = [c / total_samples for c in total_correct]

    return total_loss, total_correct

@torch.no_grad()
def test_epoch(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    for batch in loader:
        batch = batch.to(device)
        y = batch.y
        
        if batch.edge_attr is None:
            pred = model(batch.x.float(), batch.edge_index, None, tau=1000).argmax(-1)
        else:
            pred = model(batch.x.float(), batch.edge_index, batch.edge_attr.float(), tau=1000).argmax(-1)
        
        total_correct += pred.eq(y).sum().item()
        total_samples += y.size(0)
    
    acc = total_correct / total_samples
    return acc

def train_seed(dataset_name, baseline_path, args, seed, device):
    set_seed(seed)

    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    path = create_folder_logic(dataset_name, args, baseline_args, seed=seed)
    shutil.rmtree(path)
    path = create_folder_logic(dataset_name, args, baseline_args, seed=seed)

    os.mkdir(os.path.join(path, 'code'))
    for f in glob.glob('*.py'):
        shutil.copy(f, os.path.join(path, 'code'))

    with open(os.path.join(path, 'args.json'), 'w') as f:
        args_copy = {k: (v.item() if hasattr(v, 'item') else v) for k,v in args.items()}
        json.dump(args_copy, f)

    dataset = get_dataset(dataset_name)

    print(f'Training logic model on {dataset_name}')
    print(baseline_args)
    print(args)

    num_classes = dataset.num_classes
    num_features = dataset.num_features
    num_features_edge = dataset.num_edge_features
    print('Num features:', num_features, ' Num classes:', num_classes, ' Num edge features:', num_features_edge)

    if num_features == 0: num_features = 10
    
    # Load baseline split indices
    baseline_data = pickle.load(open(os.path.join(baseline_path, 'data.pkl'), 'rb'))
    train_indices = baseline_data['train_indices']
    val_indices = baseline_data['val_indices']
    test_indices = baseline_data['test_indices']

    # Create subsets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)

    with open(os.path.join(path, 'data.pkl'), 'wb') as f:
        pickle.dump({
            'dataset': dataset_name,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices
        }, f)

    # Load baseline model
    model = GIN(num_features=num_features, num_features_edge=num_features_edge, num_classes=num_classes, 
                hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers'], 
                nogumbel=baseline_args['nogumbel'], edge_once=baseline_args['edge_once'], 
                layer_double=baseline_args['layer_double']).to(device)
    model.load_state_dict(torch.load(os.path.join(baseline_path, 'best.pt'), map_location='cpu'))
    model.eval()
    for p in model.parameters():
        p.requires_grad_ = False
    
    baseline_test_acc = test_epoch(model, test_loader, device)
    print('Baseline Test Acc:', baseline_test_acc)
    

    model_tell = GINTELL(num_features=num_features, num_features_edge=num_features_edge, num_classes=num_classes, 
                             hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers'],
                             negative_concatenate=args['negative_concatenate'],
                             edge_again=args['edge_again']).to(device)
    
    optimizer = torch.optim.AdamW(model_tell.parameters(), lr=args['lr'], weight_decay=args['l2'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=300, min_lr=1e-5, verbose=True)
    
    # Training loop
    patience = 0
    max_patience = 900
    best_val_acc = 0
    best_test_acc = 0
    train_accs = []
    val_accs = []
    test_accs = []
    
    for epoch in range(args['epochs']):
        train_loss, train_acc = train_epoch(model, model_tell, train_loader, device, optimizer, num_classes, 
                                           train_full=epoch>=args['warmup_epochs'], 
                                           conv_reg=args['conv_reg'], fc_reg=args['fc_reg'], 
                                           only_teacher=args['only_teacher'])
        val_acc = test_epoch(model_tell, val_loader, device)
        test_acc = test_epoch(model_tell, test_loader, device)

        if epoch>=args['warmup_epochs']:
            scheduler.step(val_acc)
            patience += 1

        if epoch>=args['warmup_epochs'] and val_acc >= best_val_acc:
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
            print('Early stopping')
            break
    
    torch.save(model_tell, os.path.join(path, 'last.pt'))
    model_tell = torch.load(os.path.join(path, 'best.pt'))

    val_acc = test_epoch(model_tell, val_loader, device)
    test_acc = test_epoch(model_tell, test_loader, device)

    results = {
        'seed': seed,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }

    return results

def eval_seed(dataset_name, baseline_path, args, seed, device):
    set_seed(seed)

    baseline_args = json.load(open(os.path.join(baseline_path, 'args.json'), 'r'))
    path = create_folder_logic(dataset_name, args, baseline_args, seed=seed)

    dataset = get_dataset(dataset_name)

    print(f'Evaluating logic model on {dataset_name}')
    print(baseline_args)
    print(args)

    num_classes = dataset.num_classes
    num_features = dataset.num_features
    num_features_edge = dataset.num_edge_features
    print('Num features:', num_features, ' Num classes:', num_classes, ' Num edge features:', num_features_edge)

    if num_features == 0: num_features = 10
    
    # Load split indices
    baseline_data = pickle.load(open(os.path.join(baseline_path, 'data.pkl'), 'rb'))
    val_indices = baseline_data['val_indices']
    test_indices = baseline_data['test_indices']
    
    from torch.utils.data import Subset
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)
    
    # Load baseline model
    model = GIN(num_features=num_features, num_features_edge=num_features_edge, num_classes=num_classes, 
                hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers'], 
                nogumbel=baseline_args['nogumbel'], edge_once=baseline_args['edge_once'], 
                layer_double=baseline_args['layer_double']).to(device)
    model.load_state_dict(torch.load(os.path.join(baseline_path, 'best.pt'), map_location='cpu'))
    model.eval()
    for p in model.parameters():
        p.requires_grad_ = False
    
    baseline_test_acc = test_epoch(model, test_loader, device)
    print('Baseline Test Acc:', baseline_test_acc)
    
    # Load GINTELL model
    model_tell = torch.load(os.path.join(path, 'best.pt'))

    val_acc = test_epoch(model_tell, val_loader, device)
    test_acc = test_epoch(model_tell, test_loader, device)

    results = {
        'seed': seed,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }

    return results


def train_eval(dataset_name, baseline_path, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline_args = json.load(open(os.path.join(baseline_path, '0', 'args.json'), 'r'))
    
    seed_todo = args.pop('seed', None)
    only_eval = args.pop('only_eval', False)
    
    path = create_folder_logic(dataset_name, args, baseline_args)

    seeds = range(SEEDS)
    if seed_todo is not None:
        seeds = [seed_todo]
    
    results = []
    if not only_eval:
        for seed in seeds:
            results.append(train_seed(dataset_name, os.path.join(baseline_path, str(seed)), args, seed, device))

    print(results)

    if only_eval or seed_todo is not None:
        results = []
        for seed in range(SEEDS):
            try:
                r = eval_seed(dataset_name, os.path.join(baseline_path, str(seed)), args, seed, device)
                results.append(r)
                print(r)
            except Exception as e: 
                print(e)

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

    parser.add_argument('--dataset',       default='AromaticCarbon', type=str,     help='Dataset to use')
    parser.add_argument('--baseline_path', default=None,       type=str,     help='Baseline path')
    parser.add_argument('--epochs',        default=1000,       type=int,     help='Epochs')
    parser.add_argument('--warmup_epochs', default=600,       type=int,     help='Epochs')
    parser.add_argument('--batch_size',    default=32,         type=int,     help='Batch Size')
    parser.add_argument('--lr',            default=0.01,      type=float,   help='Learning Rate')
    parser.add_argument('--l2',            default=0.0001,      type=float,     help='Weight decay')
    parser.add_argument('--conv_reg',      default=0.001,      type=float,   help='Conv layer regularization')
    parser.add_argument('--fc_reg',        default=0.01,      type=float,    help='Last layer regularization')
    parser.add_argument('--negative_concatenate', default=2,   type=int,    help='0: no neg, 1: first layer neg, 2: all layers neg')
    parser.add_argument('--edge_again',    action='store_true',              help='Use edge features at every layer')
    parser.add_argument('--only_teacher',    default=2, type=int,              help='Only train using the teacher model')
    parser.add_argument('--only_eval',    action='store_true',              help='Number of Convolutional Layers')
    parser.add_argument('--seed',          default=None,      type=int,    help='Number of Convolutional Layers')

    args = parser.parse_args().__dict__
    
    dataset_name = args.pop('dataset')
    baseline_path = args.pop('baseline_path')
    if baseline_path is None:
        baseline_path = get_best_baseline_path(dataset_name)
        print('Baseline path found:', baseline_path)
    train_eval(dataset_name, baseline_path, args)