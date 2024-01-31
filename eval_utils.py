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
    mode='bt',
    print_ranking=False,
    **kwargs
):
    if mode == 'bt':
        ll_fn = bt_log_likelihood
        acc_fn = bt_accuracy
    elif mode == 'rk':
        ll_fn = rk_log_likelihood
        acc_fn = rk_accuracy

    start_time = time.time()
    ratings = get_ratings_fn(train_matchups, train_outcomes, **kwargs)
    train_nll = -ll_fn(ratings, train_matchups, train_outcomes, **kwargs)
    test_nll = -ll_fn(ratings, test_matchups, test_outcomes, **kwargs)    
    print(f'train nll: {train_nll:.6f}\ttest nll: {test_nll:.6f}')
    train_acc = acc_fn(ratings, train_matchups, train_outcomes, **kwargs)
    test_acc = acc_fn(ratings, test_matchups, test_outcomes, **kwargs)
    print(f'train acc: {train_acc:.6f}\ttest acc: {test_acc:.6f}')
    mask = test_outcomes != 0.5
    test_acc_no_draw = acc_fn(ratings, test_matchups[mask], test_outcomes[mask], **kwargs)
    print(f'test acc no draw: {test_acc_no_draw:.6f}')

    metrics = {
        'train_nll' : train_nll,
        'test_nll' : test_nll,
        'train_acc' : train_acc,
        'test_acc' :  test_acc,
        'test_acc_no_draw' : test_acc_no_draw,
        'duration (s)' :  time.time() - start_time
    }
    return metrics, ratings


def eval_seed(train_df, test_df, seed=0, verbose=False, competitor_cols=['model_a', 'model_b'], outcome_col=['outcome']):
    train_matchups, train_outcomes, competitors = preprocess(train_df, competitor_cols, outcome_col)
    test_matchups, test_outcomes, _ = preprocess(test_df, competitor_cols, outcome_col)
    draw_rate = (train_outcomes == 0.5).mean()
    if verbose: print(f'{draw_rate=}')
    metrics = []

    k = 4.0
    base = 10.0
    scale = 400.0

    elo_fn = partial(get_bootstrap_elo_ratings, k=k, num_bootstrap=1, seed=seed)
    print(f'evaluating elo: {k=}, {base=}, {scale=}')
    elo_metrics, elo_ratings = evaluate(train_matchups, train_outcomes, test_matchups, test_outcomes, elo_fn, 'bt', base=base, scale=scale)
    elo_metrics['method'] = 'elo'
    metrics.append(elo_metrics)
    if verbose: print_top_k(elo_ratings, competitors)
    print('')

    num_boot = 100
    bootstrap_elo_fn = partial(get_bootstrap_elo_ratings, num_bootstrap=num_boot, k=k, seed=seed)
    print(f'evaluating bootstrap elo: {num_boot=}, {k=}, {base=}, {scale=}')
    bootstrap_elo_metrics, bootstrap_elo_ratings = evaluate(train_matchups, train_outcomes, test_matchups, test_outcomes, bootstrap_elo_fn, 'bt', base=base, scale=scale)
    bootstrap_elo_metrics['method'] = 'bootstrap elo'
    metrics.append(bootstrap_elo_metrics)
    if verbose: print_top_k(bootstrap_elo_ratings, competitors)
    print('')

    base = math.e
    scale = 1.0
    bt_fn = partial(get_bt_ratings_lbfgs, base=base, scale=scale)
    print(f'evaluating lbfgs bt {base=}, {scale=}')
    bt_metrics, bt_ratings = evaluate(train_matchups, train_outcomes, test_matchups, test_outcomes, bt_fn, 'bt',  base=base, scale=scale)
    bt_metrics['method'] = 'bt'
    metrics.append(bt_metrics)
    if verbose: print_top_k(bt_ratings, competitors)
    print('')

    theta = 2.0 # 2.0 seems good
    # max_iter = 100
    # ilsr_fn = partial(get_ilsr_ratings, theta=theta, max_iter=max_iter, eps=1e-6)
    # print(f'evaluating ilsr rk {theta=}, {max_iter=}')
    # ilsr_metrics, ilsr_ratings = evaluate(train_matchups, train_outcomes, test_matchups, test_outcomes, ilsr_fn, 'rk', theta=theta)
    # ilsr_metrics['method'] = 'ilsr'
    # metrics.append(ilsr_metrics)
    # if verbose: print_top_k(ilsr_ratings, competitors)
    # print('')

    rk_fn = get_rk_ratings_lbfgs
    print(f'evaluating rk lbfgs {theta=}')
    rk_metrics, rk_ratings = evaluate(train_matchups, train_outcomes, test_matchups, test_outcomes, rk_fn, 'rk', theta=theta)
    rk_metrics['method'] = 'rk_lbfgs'
    metrics.append(rk_metrics)
    if verbose: print_top_k(rk_ratings, competitors)
    print('')

    for metric in metrics:
        metric['seed'] = seed
    return metrics

