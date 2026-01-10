from train_logic_node_aromatic import train_eval, get_best_baseline_path
# from skopt import gp_minimize
# from skopt.space import Real, Integer, Categorical
# from skopt.utils import use_named_args
import argparse


import itertools
from joblib import Parallel, delayed
import random

class CustomGridSearch:
    def __init__(self, param_grid, scoring_function, n_jobs=1):
        self.param_grid = param_grid
        self.scoring_function = scoring_function
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = None
        self.results_ = []

    def fit(self, dataset_name):
        # Genera tutte le combinazioni
        all_combinations = list(itertools.product(*(self.param_grid[param] for param in self.param_grid)))
        random.shuffle(all_combinations)
        
        # Filtra combinazioni non valide
        valid_combinations = []
        for params in all_combinations:
            param_dict = {param: params[i] for i, param in enumerate(self.param_grid)}
            # Regola: warmup_epochs = 0 solo se only_teacher = 0
            if param_dict['warmup_epochs'] == 0 and param_dict['only_teacher'] != 0:
                continue
            valid_combinations.append(params)

        print(f"Total combinations: {len(valid_combinations)} (filtered from {len(all_combinations)})")
        
        def evaluate_params(params):
            param_dict = {param: params[i] for i, param in enumerate(self.param_grid)}
            score = self.scoring_function(dataset_name, **param_dict)
            return param_dict, score

        # Parallelizza l’esecuzione
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_params)(params) for params in valid_combinations
        )

        # Trova i migliori parametri
        for param_dict, score in results:
            self.results_.append({'params': param_dict, 'score': score})
            if self.best_score_ is None or score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = param_dict
        
        return self


    def get_results(self):
        return self.results_


params  = {
    'epochs': [1000],
    'warmup_epochs': [0, 600],
    'batch_size': [32], 
    'lr': [0.001, 0.01],
    'l2': [0.0],
    'conv_reg': [0.001],
    'fc_reg': [0.01],
    'negative_concatenate': [0, 1, 2],
    'edge_again': [False],
    'only_teacher': [0, 1, 2],
}


def scoring_function(dataset_name, **params):
    baseline_path = get_best_baseline_path(dataset_name)
    score = train_eval(dataset_name, baseline_path, params)['val_acc_mean']
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='optimize_logic_node.py')
    parser.add_argument('--dataset',  default='AromaticCarbon', type=str, help='Dataset to use')
    parser.add_argument('--n_jobs',  default=10, type=int, help='Number of jobs')
    args = parser.parse_args()
    dataset_name = args.dataset
    n_jobs = args.n_jobs

    grid_search = CustomGridSearch(params, scoring_function, n_jobs=n_jobs)
    grid_search.fit(dataset_name)
    
    # Retrieve results
    results = grid_search.get_results()
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)
    # print("All Results:", results)

    