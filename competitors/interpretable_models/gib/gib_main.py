from itertools import product

import argparse
from datasets import get_dataset
from ours_train_eval import cross_validation_with_val_set

from gib_gin import GIBGIN, Discriminator
from gib_gat import GIBGAT
from gib_sage import GIBSAGE
from gib_gcn  import GIBGCN
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)#default = 100
# parser.add_argument('--epochs', type=int, default=3)#default = 100
parser.add_argument('--batch_size', type=int, default=128)#default = 128
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--net', type=int, default=0)
# parser.add_argument('--normalize', type=bool, default=False)
parser.add_argument('--inner_loop', type=int, default=50)
# parser.add_argument('--mi_weight', type=float, default=0.0001)
# parser.add_argument('--pp_weight', type=float, default=0.0003)
parser.add_argument('--mi_weight', type=float, default=0.1)
parser.add_argument('--pp_weight', type=float, default=0.3)
parser.add_argument('--folds', type=int, default=1)
args = parser.parse_args()

layers = [5, 5, 5, 5, 5, 5, 5]
hiddens = [32, 16, 32, 32, 32, 16, 32]
batches = [128, 128, 32, 128, 32, 32, 128]
datasets = ['NCI1', 'ba_2motifs', 'PROTEINS', 'MUTAG', 'Mutagenicity', 'BBBP', 'BaMultiShapes']
#datasets = [args.dataset]
Net = GIBGIN


idx = datasets.index(args.dataset)

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))
seeds = [0,1,2,3,4]
results = []
num_layers = layers[idx]
hidden = hiddens[idx]
batch_size = batches[idx]
dataset_name = args.dataset


accs = []

best_result = (float('inf'), 0, 0)  # (loss, acc, std)
print('-----\n{} - {}'.format(dataset_name, Net.__name__))
seeds_results = {}
for seed in seeds:
    dataset = get_dataset(dataset_name, sparse=True)
    # model = Net(dataset, num_layers, hidden, args.normalize)
    model = Net(dataset, num_layers, hidden)
    discriminator = Discriminator(hidden)
    loss, acc, std, results = cross_validation_with_val_set(
        dataset,
        model,
        discriminator,
        folds=args.folds,
        seed=seed,
        epochs=args.epochs,
        batch_size=batch_size,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_step_size=args.lr_decay_step_size,
        weight_decay=0.001,
        inner_loop = args.inner_loop,
        mi_weight = args.mi_weight,
        pp_weight=args.pp_weight,
        logger= None
    )
    accs.append(acc)
    print(dataset_name, seed, acc)
    seeds_results[seed] = results
        
            
        
import pickle
pickle.dump(seeds_results, open(f'results/{dataset_name}.pkl', 'wb'))
print(dataset_name, np.mean(accs)*100, np.std(accs)*100)
