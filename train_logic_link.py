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
from model_link import GIN, GINTELL
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score
import time

SEEDS = 5


def get_best_baseline_path(dataset_name):
    l = glob.glob(f'results/{dataset_name}/*/results.json')
    fl = [json.load(open(f)) for f in l]
    df = pd.DataFrame(fl)
    if df.shape[0] == 0: return None
    df['fname'] = l
    df = df.sort_values(by=['val_auc_mean', 'val_auc_std', 'test_auc_std'], ascending=[True,False,False])
    df = df[df.fname.str.contains('nogumbel=False')]
    fname = df.iloc[-1]['fname']
    fname = fname.replace('/results.json', '')
    return fname

def train_epoch(model, model_tell, loader, device, optimizer, train_full=True, conv_reg=0.001, fc_reg=0.01, only_teacher=0):
    model_tell.train()

    total_loss = [0]*(len(model_tell.convs)+1)
    total_correct = [0]*(len(model_tell.convs)+1)


    if train_full:
        if model_tell.phi_node is not None:
            model_tell.phi_node.tau = 10
        if model_tell.phi_edge is not None:
            model_tell.phi_edge.tau = 10
        for layer in model_tell.convs:
            layer.nn_1.phi_in.tau = 10

    n_batches = 0

    for data in loader:
        n_batches += 1
        data = data.to(device)
    
        optimizer.zero_grad()

        with torch.no_grad():
            if data.edge_attr is None:
                layers_x, layers_y, layers_y_interm = model.forward_e(data.x.float(), data.edge_index, None, data.edge_label_index, tau=1000)
            else:
                layers_x, layers_y, layers_y_interm = model.forward_e(data.x.float(), data.edge_index, data.edge_attr.float(), data.edge_label_index, tau=1000)

        loss = 0
        last_layer_out = None

        if train_full == False or (train_full == True and  (only_teacher == 1 or only_teacher == 2)):

            if data.edge_attr is not None and model_tell.negative_concatenate!=0:
                if model_tell.phi_edge is not None:
                    edge_attr = data.edge_attr.float()
                    edge_attr = model_tell.phi_edge(edge_attr)
                    edge_attr = torch.hstack([edge_attr, 1-edge_attr]).float()
                else:
                    edge_attr = torch.hstack([data.edge_attr, 1-data.edge_attr]).float()

            elif data.edge_attr is not None and model_tell.negative_concatenate==0:
                if model_tell.phi_edge is not None:
                    edge_attr = data.edge_attr.float()
                    edge_attr = model_tell.phi_edge(edge_attr).float()
                else:
                    edge_attr = data.edge_attr.float()

            for i, (layer, layer_x, layer_y, layer_y_interm) in enumerate(zip(model_tell.convs, layers_x[:-1], layers_y[:-1], layers_y_interm)):
                    
                if i==0 and not model_tell.input_binary:
                    layer_x = model_tell.phi_node(layer_x)
                    loss_binarization_input = conv_reg*(model_tell.phi_node.entropy + model_tell.phi_node.reg_loss)
                    print('Input binarization loss:', loss_binarization_input.item())
                    loss += loss_binarization_input

                if i==0 and not model_tell.edge_binary:
                    loss_binarization_edge = conv_reg*(model_tell.phi_edge.entropy + model_tell.phi_edge.reg_loss)
                    print('Edge binarization loss:', loss_binarization_edge.item())
                    loss += loss_binarization_edge
                
                if (i==0 and model_tell.negative_concatenate==1) or model_tell.negative_concatenate==2:
                    layer_x = torch.hstack([layer_x, 1-layer_x])

                if i == 0 and data.edge_attr is not None:
                    layer_out, layer_out_interm = layer(layer_x, data.edge_index.to(device), edge_attr)
                elif i!=0 and data.edge_attr is not None and model_tell.edge_again:
                    layer_out, layer_out_interm = layer(layer_x, data.edge_index.to(device), edge_attr)
                else:
                    layer_out, layer_out_interm = layer(layer_x, data.edge_index.to(device), None)
                
                layer_loss = F.binary_cross_entropy(layer_out.reshape(-1), layer_y.reshape(-1)) + conv_reg* (layer.nn_1.reg_loss + layer.nn_1.entropy_loss)
                layer_loss += F.binary_cross_entropy(layer_out_interm.reshape(-1), layer_y_interm.reshape(-1)) + conv_reg* (layer.nn_0.reg_loss + layer.nn_0.entropy_loss)
                #print(f'Layer {i} loss teacher:', layer_loss.item())
                loss += layer_loss

                if train_full and i!=0:
                    layer_x = last_layer_out

                    if model_tell.negative_concatenate == 2:
                        layer_x = torch.hstack([layer_x, 1-layer_x])
                    
                    if data.edge_attr is not None and model_tell.edge_again:
                        layer_out, layer_out_interm = layer(layer_x, data.edge_index.to(device), edge_attr)
                    else:
                        layer_out, layer_out_interm = layer(layer_x, data.edge_index.to(device), None)
                    
                    
                    layer_loss = F.binary_cross_entropy(layer_out.reshape(-1), layer_y.reshape(-1)) + conv_reg* (layer.nn_1.reg_loss + layer.nn_1.entropy_loss)
                    layer_loss += F.binary_cross_entropy(layer_out_interm.reshape(-1), layer_y_interm.reshape(-1)) + conv_reg* (layer.nn_0.reg_loss + layer.nn_0.entropy_loss)
                    #print(f'Layer {i} loss teacher full training:', layer_loss.item())
                    loss += layer_loss

                last_layer_out = layer_out
                total_loss[i] += layer_loss.item()/len(loader)
                total_correct[i] += ((layer_out >= 0.5).long() == layer_y).sum().item() / (layer_y.shape[-2]*layer_y.shape[-1]*len(loader))

            layer_x = layers_x[-1]
            if model_tell.negative_concatenate==2:
                layer_x = torch.hstack([layer_x, 1-layer_x])
            out = model_tell.fc(layer_x)
            

            if only_teacher == 2 or only_teacher == 0:
                layer_y = layers_y[-1]
                layer_y_true = layer_y.view(-1)
                loss_fc_teacher = F.binary_cross_entropy(out.view(-1), layer_y_true)
            else:
                loss_fc_teacher = F.binary_cross_entropy(out.view(-1), data.edge_label.float())
            #print('FC layer loss teacher:', loss_fc_teacher.item())
            loss += loss_fc_teacher

        if train_full:
            if data.edge_attr is not None:
                out = model_tell(data.x.float().to(device), data.edge_index.to(device), data.edge_attr.float().to(device), data.edge_label_index.to(device))
            else:
                out = model_tell(data.x.float().to(device), data.edge_index.to(device), None, data.edge_label_index.to(device))
            

            loss_fc_teacher_full= F.binary_cross_entropy(out.view(-1), data.edge_label.float()) 
            #print('FC layer loss teacher full training:', loss_fc_teacher_full.item())
            loss += loss_fc_teacher_full

        loss_fc_reg= fc_reg*(model_tell.fc.reg_loss + model_tell.fc.entropy_loss)
        #print('FC layer regularization loss:', loss_fc_reg.item())
        loss += loss_fc_reg
        loss.backward()
        zero_nan_gradients(model_tell)
        optimizer.step()
        total_loss[-1] += loss.item() / len(loader)
        total_correct[-1] += roc_auc_score(data.edge_label.float().cpu().numpy(), out.detach().cpu().numpy())

    total_correct[-1] /= n_batches
    return total_loss, total_correct


@torch.no_grad()
def test_epoch(model, data, device):
    model.eval()
    data = data.to(device)

    if data.edge_attr is None:
        out = model(data.x.float(), data.edge_index, None, data.edge_label_index, tau=1000)
    else:
        out = model(data.x.float(), data.edge_index, data.edge_attr.float(), data.edge_label_index, tau=1000)

    auc = roc_auc_score(data.edge_label.float().cpu().numpy(), out.cpu().numpy())
    
    return auc

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
        args = {k: (v.item() if hasattr(v, 'item') else v) for k,v in args.items()}
        json.dump(args, f)


    dataset = get_dataset(dataset_name)

    print(f'Training logic model on {dataset_name}')
    print(baseline_args)
    print(args)

    num_features = dataset.num_features
    num_features_edge = dataset.num_edge_features
    print('Num features:', num_features, ' Num edge features:', num_features_edge)

    if num_features == 0: num_features = 10
    
    data = pickle.load(open(os.path.join(baseline_path, 'data.pkl'), 'rb'))
    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']

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

    with open(os.path.join(path, 'data.pkl'), 'wb') as f:
        pickle.dump({
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data
        }, f)

    model = GIN(num_features=num_features, num_features_edge=num_features_edge, hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers'], nogumbel=baseline_args['nogumbel'],
                edge_once=baseline_args['edge_once'], layer_double=baseline_args['layer_double']).to(device)
    model.load_state_dict(torch.load(os.path.join(baseline_path, 'best.pt'), map_location='cpu'))
    model.eval()
    for p in model.parameters():
        p.requires_grad_ = False
    print('Baseline Acc:', test_epoch(model, test_data, device))
    
    model_tell = GINTELL(num_features=num_features, num_features_edge=num_features_edge, hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers'],
                              negative_concatenate=args['negative_concatenate'],
                              edge_again=args['edge_again']).to(device)
    
    optimizer = torch.optim.AdamW(model_tell.parameters(), lr=args['lr'], weight_decay=args['l2'])
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.99, patience=300, min_lr=1e-5, verbose=True)
    
    # Training loop
    patience = 0
    max_patience = 900
    best_val_auc = 0
    best_test_auc = 0
    train_aucs = []
    val_aucs = []
    test_aucs = []

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[10]*baseline_args['num_layers'],
        neg_sampling_ratio=2.0,
        batch_size=128,
        shuffle=True,
    )

    for epoch in range(args['epochs']):

        train_loss, train_auc= train_epoch(model, model_tell, train_loader, device, optimizer, train_full=epoch>=args['warmup_epochs'], conv_reg=args['conv_reg'], fc_reg=args['fc_reg'], only_teacher=args['only_teacher'])
        val_auc = test_epoch(model_tell, val_data, device)
        test_auc = test_epoch(model_tell, test_data, device)
        val_auc_teacher = test_epoch(model, val_data, device)
        test_auc_teacher = test_epoch(model, test_data, device)

        if epoch>=args['warmup_epochs']:
            scheduler.step(val_auc)
            patience += 1

        if epoch>=args['warmup_epochs'] and val_auc >= best_val_auc:
            torch.save(model_tell, os.path.join(path, 'best.pt'))
            best_val_auc = val_auc
            best_test_auc = test_auc
            patience = 0
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss}, Train AUC: {train_auc}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}, Val AUC Teacher: {val_auc_teacher:.4f}, Test AUC Teacher: {test_auc_teacher:.4f}')
            print(f'\t\t Best Val AUC: {best_val_auc:.4f}, Best Test AUC: {best_test_auc:.4f}')

        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        test_aucs.append(test_auc)

        if patience >= max_patience:
            print('Early stopping')
            break
    
    torch.save(model_tell, os.path.join(path, 'last.pt'))
    # model_tell.load_state_dict(torch.load(os.path.join(path, 'best.pt')))
    model_tell = torch.load(os.path.join(path, 'best.pt'))

    val_auc = test_epoch(model_tell, val_data, device)
    test_auc = test_epoch(model_tell, test_data, device)

    results = {
        'seed': seed,
        'val_auc': val_auc,
        'test_auc': test_auc,
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

    num_features = dataset.num_features
    num_features_edge = dataset.num_edge_features
    print('Num features:', num_features, ' Num edge features:', num_features_edge)

    if num_features == 0: num_features = 10
    
    data = pickle.load(open(os.path.join(baseline_path, 'data.pkl'), 'rb'))

    train_data = data['train_data']
    val_data = data['val_data']
    test_data = data['test_data']
    
    model = GIN(num_features=num_features, num_features_edge=num_features_edge, hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers'], nogumbel=baseline_args['nogumbel'],
                edge_once=baseline_args['edge_once'], layer_double=baseline_args['layer_double']).to(device)
    model.load_state_dict(torch.load(os.path.join(baseline_path, 'best.pt'), map_location='cpu'))
    model.eval()
    for p in model.parameters():
        p.requires_grad_ = False
    print('Baseline Acc:', test_epoch(model, test_data, device))
    
    model_tell = GINTELL(num_features=num_features, num_features_edge=num_features_edge, hidden_dim=baseline_args['hidden_dim'], num_layers=baseline_args['num_layers'],
                              negative_concatenate=args['negative_concatenate'],
                              edge_again=args['edge_again']).to(device)
    model_tell = torch.load(os.path.join(path, 'best.pt'))

    val_acc = test_epoch(model_tell, val_data, device)
    test_acc = test_epoch(model_tell, test_data, device)

    results = {
        'seed': seed,
        'val_auc': val_acc,
        'test_auc': test_acc,
    }

    return results


def train_eval(dataset_name, baseline_path, args):
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
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
            start_time = time.time()
            result = train_seed(dataset_name, os.path.join(baseline_path, str(seed)), args, seed, device)
            result['training_time'] = (time.time() - start_time) / 60
            results.append(result)

    print(results)

    if only_eval or seed_todo is not None:
        results = []
        for seed in range(SEEDS):
            try:
                r = eval_seed(dataset_name, os.path.join(baseline_path, str(seed)), args, seed, device)
                results.append(r)
                print(r)
            except Exception as e: print(e)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(path, 'total_results.csv'))

    ret = {
        'val_auc_mean': df['val_auc'].mean(),
        'test_auc_mean': df['test_auc'].mean(),
        'val_auc_std': df['val_auc'].std(),
        'test_auc_std': df['test_auc'].std(),
        'training_time_mean_minutes': df['training_time'].mean(),
        'training_time_std_minutes': df['training_time'].std()
    }

    with open(os.path.join(path, 'results.json'), 'w') as f:
        json.dump(ret, f)

    return ret
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train_baseline.py')

    parser.add_argument('--dataset',       default='Cora', type=str,     help='Dataset to use')
    parser.add_argument('--baseline_path', default=None,       type=str,     help='Baseline path')
    parser.add_argument('--epochs',        default=6000,       type=int,     help='Epochs')
    parser.add_argument('--warmup_epochs', default=3000,       type=int,     help='Epochs')
    parser.add_argument('--lr',            default=0.001,      type=float,   help='Learning Rate')
    parser.add_argument('--l2',            default=0.1,      type=float,     help='Weight decay')
    parser.add_argument('--conv_reg',      default=0.001,      type=float,   help='Conv layer regularization')
    parser.add_argument('--fc_reg',        default=0.1,      type=float,    help='Last layer regularization')
    parser.add_argument('--negative_concatenate', default=2,   type=int,    help='0: no neg, 1: first layer neg, 2: all layers neg')
    parser.add_argument('--edge_again',    action='store_true',              help='Use edge features at every layer')
    parser.add_argument('--only_teacher',    default=1, type=int,              help='Only train using the teacher model')
    parser.add_argument('--only_eval',    action='store_true',              help='Number of Convolutional Layers')
    parser.add_argument('--seed',          default=None,      type=int,    help='Number of Convolutional Layers')

    args = parser.parse_args().__dict__
    
    dataset_name = args.pop('dataset')
    baseline_path = args.pop('baseline_path')
    if baseline_path is None:
        baseline_path = get_best_baseline_path(dataset_name)
        print('Baseline path found:', baseline_path)
    train_eval(dataset_name, baseline_path, args)

    