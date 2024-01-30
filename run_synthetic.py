import time
import math
from functools import partial
import numpy as np
import pandas as pd
from data_utils import split, preprocess, print_top_k
from likelihoods import bt_log_likelihood, rk_log_likelihood
from bradley_terry_models import get_bt_ratings_lbfgs
from rao_kupper_models import get_rao_kupper_ratings, get_rk_ratings_lbfgs
from luce_models import get_ilsr_ratings
from elo import get_elo_ratings, get_bootstrap_elo_ratings
from metrics import bt_accuracy, rk_accuracy
from riix.utils import generate_matchup_data
from eval_utils import eval_seed


if __name__ == '__main__':
    seed = 0
    competitor_cols = ['competitor_1', 'competitor_2']
    outcome_col = 'outcome'


    metrics = []
    num_seeds = 25
    for seed in range(num_seeds):
        df = generate_matchup_data(
            num_matchups=10000,
            num_competitors=50,
            num_rating_periods=100,
            skill_sd=1.0,
            outcome_noise_sd=2.0,
            draw_margin=0.2,
            seed=seed
        )
        draw_rate = (df['outcome'] == 0.5).mean()
        print(f'{draw_rate=}')
        train_df, test_df = split(df, test_size=0.2, shuffle=True, seed=seed)
        seed_metrics = eval_seed(
            train_df,
            test_df,
            seed=seed,
            verbose=False,
            competitor_cols=competitor_cols,
            outcome_col=outcome_col
        )
        metrics.extend(seed_metrics)
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.groupby(['method']).mean())