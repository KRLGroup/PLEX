from utils import *
import torch
import numpy as np
import pandas as pd
import glob
import json
import os
import pickle
from torch_geometric.loader import DataLoader
from train_logic import test_epoch

def get_best_path(dataset_name):
    l = glob.glob(f'results_logic/{dataset_name}/*/*/results.json')
    fl = [json.load(open(f)) for f in l]
    df = pd.DataFrame(fl)
    if df.shape[0] == 0: return None
    df['fname'] = l
    df = df.sort_values(by=['val_acc_mean', 'val_acc_std', 'test_acc_std'], ascending=[True,False,False])
    df = df[df.fname.str.contains('nogumbel=False')]
    print(df.tail())
    fname = df.iloc[-1]['fname']
    fname = fname.replace('/results.json', '')
    return fname

dataset_names = ['Mutagenicity', 'PROTEINS']
seeds = [0,1,2,3,4]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

diz = {}

for dataset_name in dataset_names:
    diz[dataset_name] = {}
    for seed in seeds:
        diz[dataset_name][seed] = {}
        save_base_dir = f'explanations/{dataset_name}/{seed}'
        results_path = os.path.join(get_best_path(dataset_name), str(seed))

        data = pickle.load(open(os.path.join(results_path, 'data.pkl'), 'rb'))

        #test_data = data['test_data']
        dataset = get_dataset(dataset_name)
        # #Graph Classification
        test_dataset = dataset[data['test_indices']]
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model_tell_before = torch.load(os.path.join(results_path, 'best.pt'), map_location=device)
        model_tell_after = torch.load(os.path.join(save_base_dir, 'model_tell_final.pt'), map_location=device)

        # before_acc = test_epoch(model_tell_before, dataset.data, dataset.data.test_mask, device)
        # after_acc= test_epoch(model_tell_after, dataset.data, dataset.data.test_mask, device)
        # before_acc = test_epoch(model_tell_before, test_data, device)
        # after_acc= test_epoch(model_tell_after, test_data, device)
        before_acc = test_epoch(model_tell_before, test_loader, device)
        after_acc= test_epoch(model_tell_after, test_loader, device)

        diz[dataset_name][seed]['before_acc'] = before_acc
        diz[dataset_name][seed]['after_acc'] = after_acc
        
        n_w_before = 0
        n_w_after = 0

        n_w_fc_before = (model_tell_before.fc.weight>1e-4).sum().item()
        n_w_fc_after = (model_tell_after.fc.weight>1e-4).sum().item()

        n_w_before += n_w_fc_before
        n_w_after += n_w_fc_after
        diz[dataset_name][seed]['fc'] = (n_w_fc_before, n_w_fc_after)

        for l in range(len(model_tell_before.convs)):
            n_w_0_before = (model_tell_before.convs[l].nn_0.weight>1e-4).sum().item()
            n_w_1_before = (model_tell_before.convs[l].nn_1.weight>1e-4).sum().item() 
            n_w_0_after = (model_tell_after.convs[l].nn_0.weight>1e-4).sum().item()
            n_w_1_after = (model_tell_after.convs[l].nn_1.weight>1e-4).sum().item() 
            diz[dataset_name][seed][f'conv_{l}_0'] = (n_w_0_before, n_w_0_after)
            diz[dataset_name][seed][f'conv_{l}_1'] = (n_w_1_before, n_w_1_after)
            n_w_before += n_w_0_before + n_w_1_before
            n_w_after += n_w_0_after + n_w_1_after

        diz[dataset_name][seed]['total'] = (n_w_before, n_w_after)

rows = []

for dataset, seeds_dict in diz.items():

    # ---------- ACCURACY ----------
    before_acc_vals = []
    after_acc_vals = []

    for seed in seeds_dict:
        before_acc_vals.append(seeds_dict[seed]['before_acc'])
        after_acc_vals.append(seeds_dict[seed]['after_acc'])

    rows.append({
        'dataset': dataset,
        'metric': 'accuracy',
        'layer': 'total',
        'before_mean': np.mean(before_acc_vals)*100,
        'before_std': np.std(before_acc_vals)*100,
        'after_mean': np.mean(after_acc_vals)*100,
        'after_std': np.std(after_acc_vals)*100,
    })

    # ---------- WEIGHTS ----------
    weight_keys = [
        k for k in seeds_dict[next(iter(seeds_dict))]
        if k not in ['before_acc', 'after_acc']
    ]

    for key in weight_keys:
        before_vals = []
        after_vals = []

        for seed in seeds_dict:
            b, a = seeds_dict[seed][key]
            before_vals.append(b)
            after_vals.append(a)

        rows.append({
            'dataset': dataset,
            'metric': 'weights',
            'layer': key,
            'before_mean': np.mean(before_vals),
            'before_std': np.std(before_vals),
            'after_mean': np.mean(after_vals),
            'after_std': np.std(after_vals),
        })

df_stats = pd.DataFrame(rows)
df_stats.to_csv('weight_and_accuracy_stats_per_dataset.csv', index=False)

print(df_stats)

