import math
from copy import deepcopy
from tqdm import tqdm
from functools import partial
import numpy as np
import pandas as pd
from data_utils import split, generate_matchup_data
from eval_utils import eval_seed
from elo import get_bootstrap_elo_ratings
from bradley_terry_models import get_bt_ratings_lbfgs
from rao_kupper_models import get_rk_ratings_lbfgs


if __name__ == '__main__':

    data_generation_args = {
        'num_matchups': 5000,
        'num_competitors': 100,
        'strength_var': 1.0,
        'strength_noise_var': [0.25, 0.5, 1.0, 1.5, 2.0],
        'theta' : 1.0,
    }
    x_axis_key = 'strength_noise_var'

    competitor_cols = ['competitor_1', 'competitor_2']
    outcome_col = 'outcome'

    elo_config = {
        'model': 'Elo',
        'function': get_bootstrap_elo_ratings,
        'args': {
            'k': 4.0,
            'base': 10.0,
            'scale': 400.0,
            'num_bootstrap': 1,
            'likelihood': 'bt'
        }
    }

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

    rao_kupper_config = {
        'model': 'Rao Kupper',
        'function': get_rk_ratings_lbfgs,
        'args': {
            'theta': 2.0,
            'likelihood': 'rk'
        }
    }
    model_configs = [elo_config, bootstrap_elo_config, bradley_terry_config, rao_kupper_config]

    metrics = []
    num_seeds = 50
    y_axis_vals = []
    for x_axis_val in data_generation_args[x_axis_key]:
        for seed in range(num_seeds):
            seed_config = deepcopy(data_generation_args)
            seed_config['seed'] = seed
            seed_config[x_axis_key] = x_axis_val

            df = generate_matchup_data(**seed_config)
            train_df, test_df = split(df, test_size=0.2, shuffle=True, seed=seed)

            seed_metrics = eval_seed(
                train_df,
                test_df,
                model_configs=model_configs,
                seed=seed,
                verbose=False,
                competitor_cols=competitor_cols,
                outcome_col=outcome_col
            )
            metrics.extend(seed_metrics)
        metrics_df = pd.DataFrame(metrics)
        custom_order = [config['model'] for config in model_configs]
        metrics_df = metrics_df.groupby('model').mean().reindex(custom_order)
        print(metrics_df)
        y_axis_vals.append(metrics_df['test_acc'])

    plt.plot(config[x_axis_key], y_axis_vals)
    plt.xlabel(x_axis_key)
    plt.ylabel('test_acc')
    plt.legend(metrics_df.index)
    plt.show()

    