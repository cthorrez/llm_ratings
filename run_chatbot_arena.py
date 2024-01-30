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
from eval_utils import eval_seed


if __name__ == '__main__':
    # df = pd.read_json('chatbot_arena_hf.json', lines=True).drop_duplicates()
    # df = pd.read_json('chatbot_arena_12-06-2023.json', lines=True).drop_duplicates()
    # df = pd.read_json('chatbot_arena_01-06-2024.json', lines=True).drop_duplicates()
    split_seed = 0
    df = pd.read_json('chatbot_arena_01-26-2024.json', lines=True).drop_duplicates()
    train_df, test_df = split(df, test_size=0.2, shuffle=True, seed=split_seed)

    metrics = []
    num_seeds = 25
    for seed in range(num_seeds):
        seed_metrics = eval_seed(train_df, test_df, seed=seed, verbose=False)
        metrics.extend(seed_metrics)
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.groupby(['method']).mean())