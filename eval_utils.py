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
from elo import get_bootstrap_elo_ratings
from metrics import bt_accuracy, rk_accuracy



def evaluate(
    train_matchups,
    train_outcomes,
    test_matchups,
    test_outcomes,
    get_ratings_fn,
    likelihood='bt',
    print_ranking=False,
    **kwargs
):
    if likelihood == 'bt':
        ll_fn = bt_log_likelihood
        acc_fn = bt_accuracy
    elif likelihood == 'rk':
        ll_fn = rk_log_likelihood
        acc_fn = rk_accuracy

    start_time = time.time()
    ratings = get_ratings_fn(train_matchups, train_outcomes, **kwargs)
    train_nll = -ll_fn(ratings, train_matchups, train_outcomes, **kwargs)
    test_nll = -ll_fn(ratings, test_matchups, test_outcomes, **kwargs)    
    train_acc = acc_fn(ratings, train_matchups, train_outcomes, **kwargs)
    test_acc = acc_fn(ratings, test_matchups, test_outcomes, **kwargs)
    mask = test_outcomes != 0.5
    test_acc_no_draw = acc_fn(ratings, test_matchups[mask], test_outcomes[mask], **kwargs)

    metrics = {
        'train_nll' : train_nll,
        'test_nll' : test_nll,
        'train_acc' : train_acc,
        'test_acc' :  test_acc,
        'test_acc_no_draw' : test_acc_no_draw,
        'duration (s)' :  time.time() - start_time
    }
    return metrics, ratings


def eval_seed(
    train_df,
    test_df,
    model_configs,
    seed=0,
    competitor_cols=['model_a', 'model_b'],
    outcome_col=['outcome'],
    verbose=False,
    ):
    train_matchups, train_outcomes, competitors = preprocess(train_df, competitor_cols, outcome_col)
    test_matchups, test_outcomes, _ = preprocess(test_df, competitor_cols, outcome_col)
    draw_rate = (train_outcomes == 0.5).mean()
    if verbose: print(f'{draw_rate=}')
    metrics = []

    for model_config in model_configs:
        model_metrics, model_ratings = evaluate(
            train_matchups,
            train_outcomes,
            test_matchups,
            test_outcomes,
            get_ratings_fn=model_config['function'],
            **model_config['args']
        )
        model_metrics['model'] = model_config['model']
        metrics.append(model_metrics)

    for metric in metrics:
        metric['seed'] = seed
    return metrics
