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
    # df = pd.read_json('chatbot_arena_hf.json', lines=True).drop_duplicates()
    # df = pd.read_json('chatbot_arena_12-06-2023.json', lines=True).drop_duplicates()
    # df = pd.read_json('chatbot_arena_01-06-2024.json', lines=True).drop_duplicates()
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
    # metric_keys = ['train_acc', 'test_acc']
    metric_keys = ['train_nll', 'test_nll']
    thetas = np.linspace(start=1.01, stop=2.5, num=20)
    ys = []
    for theta in tqdm(thetas):
        current_config = deepcopy(rao_kupper_config)
        current_config['args']['theta'] = theta
        metrics = eval_seed(train_df, test_df, [current_config], seed=seed, verbose=False)[0]
        
        ys.append([metrics[metric_key] for metric_key in metric_keys])
    
    plt.plot(thetas, ys)
    plt.xlabel('theta')
    # plt.ylabel(metric)
    plt.legend(metric_keys)
    plt.show()



