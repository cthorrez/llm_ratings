import math
from copy import deepcopy
from tqdm import tqdm
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import split
from eval_utils import eval_seed
from rao_kupper_models import get_rk_ratings_lbfgs

if __name__ == '__main__':
    split_seed = 0
    df = pd.read_json('chatbot_arena_01-26-2024.json', lines=True).drop_duplicates()
    train_df, test_df = split(df, test_size=0.2, shuffle=True, seed=split_seed)

    rao_kupper_config = {
        'model': 'Rao Kupper',
        'function': get_rk_ratings_lbfgs,
        'args': {
            'likelihood': 'rk'
        }
    }
    seed = 0
    max_metric = True
    metric_keys = ['train_acc', 'test_acc']
    # max_metric = False
    # metric_keys = ['train_nll', 'test_nll']
    thetas = np.linspace(start=1.0001, stop=2.4, num=20)
    ys = []
    for theta in tqdm(thetas):
        current_config = deepcopy(rao_kupper_config)
        current_config['args']['theta'] = theta
        metrics = eval_seed(train_df, test_df, [current_config], seed=seed, verbose=False)[0]
        ys.append([metrics[metric_key] for metric_key in metric_keys])
    ys = np.array(ys)

    # lol that's hacky as hell but I stand by it
    # (-1)^False = (-1)^0 = 1
    # (-1)^True = (-1)^1 = -1
    # thus it switches the sign if max_metric=True so it's an argmax :D
    train_optimal_idx = np.argmin(ys[:,0] * (-1)**max_metric)
    train_optimal_theta = thetas[train_optimal_idx]
    train_optimal_test_metric = ys[:,1][train_optimal_idx]
    print(f'{train_optimal_theta=}')
    print(f'{train_optimal_test_metric=}')
    
    plt.plot(thetas, ys)
    plt.xlabel('theta')
    # plt.ylabel(metric)
    plt.legend(metric_keys)
    plt.show()



