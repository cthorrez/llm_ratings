import math
from copy import deepcopy
from tqdm import tqdm
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import split
from eval_utils import eval_seed, eval_seeds
from data_utils import split, generate_matchup_data
from elo import get_bootstrap_elo_ratings
from bradley_terry_models import get_bt_ratings_lbfgs
from rao_kupper_models import get_rk_ratings_lbfgs

if __name__ == '__main__':

    df = generate_matchup_data(
        num_matchups=10000,
        num_competitors=100,
        strength_var=1.0,
        strength_noise_var=0.1,
        theta=2.18,
        seed=0
    )
    print(len(df))
    exit(1)


    seed = 0

    rao_kupper_config = {
        'model': 'Rao Kupper',
        'function': get_rk_ratings_lbfgs,
        'args': {
            'likelihood': 'rk'
        }
    }
    thetas_fit = np.linspace(start=1.001, stop=4.0, num=25, endpoint=True)

    bootstrap_elo_config = {
        'model': 'Bootstrap Elo',
        'function': get_bootstrap_elo_ratings,
        'args': {
            'k': 4.0,
            'base': 10.0,
            'scale': 400.0,
            'num_bootstrap': 100,
            'likelihood': 'bt'
        }
    }
    bradley_terry_config = {
        'model': 'Bradley Terry',
        'function': get_bt_ratings_lbfgs,
        'args': {
            'base': math.e,
            'scale': 1.0,
            'likelihood': 'bt'
        }
    }
    baseline_configs = [bootstrap_elo_config, bradley_terry_config]

    competitor_cols = ['competitor_1', 'competitor_2']
    theta_gen = 2.18
    metric_key = 'test_acc'
    # metric_key = 'test_nll'
    ys = []

    num_seeds = 5
    for seed in range(num_seeds):
        seed_ys = []
        print('yooooooooooo')
        df = generate_matchup_data(
            num_matchups=10000,
            num_competitors=100,
            strength_var=1.0,
            strength_noise_var=0.1,
            theta=2.18,
            seed=0
        )
        true_draw_rate = (df['outcome'] == 0.5).mean()
        print(f'{true_draw_rate=}')
        train_df, test_df = split(df, test_size=0.2, shuffle=True, seed=seed)

        baseline_metrics = eval_seed(train_df, test_df, baseline_configs, seed=seed, verbose=False, competitor_cols=competitor_cols,)
        for baseline in baseline_metrics:
            seed_ys.append([baseline[metric_key]] * len(thetas_fit))

        seed_ys.append([])
        for theta_fit in tqdm(thetas_fit):
            current_rk_config = deepcopy(rao_kupper_config)
            current_rk_config['args']['theta'] = theta_fit

            current_metrics = eval_seed(train_df, test_df, [current_rk_config], seed=0, verbose=False, competitor_cols=competitor_cols)[0]
            seed_ys[-1].append(current_metrics[metric_key])
    
        seed_ys = np.array(seed_ys).T
        ys.append(seed_ys)

    ys = np.array(ys).mean(axis=0)

    plt.plot(thetas_fit, ys)
    plt.xlabel('theta')
    # plt.ylabel(metric)
    plt.legend([bc['model'] for bc in baseline_configs] + ['Rao Kupper'])
    plt.show()



