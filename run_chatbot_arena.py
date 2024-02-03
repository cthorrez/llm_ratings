import math
from tqdm import tqdm
from functools import partial
import numpy as np
import pandas as pd
from data_utils import split
from eval_utils import eval_seed
from elo import get_bootstrap_elo_ratings
from bradley_terry_models import get_bt_ratings_lbfgs
from rao_kupper_models import get_rk_ratings_lbfgs

if __name__ == '__main__':
    # df = pd.read_json('chatbot_arena_hf.json', lines=True).drop_duplicates()
    # df = pd.read_json('chatbot_arena_12-06-2023.json', lines=True).drop_duplicates()
    # df = pd.read_json('chatbot_arena_01-06-2024.json', lines=True).drop_duplicates()
    split_seed = 0
    df = pd.read_json('chatbot_arena_01-26-2024.json', lines=True).drop_duplicates()
    true_draw_rate = (df['outcome'] == 0.5).mean()
    print(f'{true_draw_rate=}')
    train_df, test_df = split(df, test_size=0.2, shuffle=True, seed=split_seed)

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
            'theta': 1.7,
            'likelihood': 'rk'
        }
    }
    model_configs = [elo_config, bootstrap_elo_config, bradley_terry_config, rao_kupper_config]

    metrics = []
    num_seeds = 1
    for seed in tqdm(range(num_seeds)):
        seed_metrics = eval_seed(train_df, test_df, model_configs, seed=seed, verbose=False)
        metrics.extend(seed_metrics)
    metrics_df = pd.DataFrame(metrics)

    custom_order = [config['model'] for config in model_configs]
    metrics_df = metrics_df.groupby('model').mean().reindex(custom_order)
    print(metrics_df)
