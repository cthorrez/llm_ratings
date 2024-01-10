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


def bt_eval(
    train_matchups,
    train_outcomes,
    test_matchups,
    test_outcomes,
    get_ratings_fn,
    base=math.e,
    scale=1.0,
    print_ranking=False,
):
    ratings = get_ratings_fn(train_matchups, train_outcomes, base=base, scale=scale)
    train_nll = -bt_log_likelihood(ratings, train_matchups, train_outcomes, base=base, scale=scale)
    test_nll = -bt_log_likelihood(ratings, test_matchups, test_outcomes, base=base, scale=scale)    
    print(f'train nll: {train_nll:.6f}\ttest nll: {test_nll:.6f}')
    train_acc = bt_accuracy(ratings, train_matchups, train_outcomes, base=base, scale=scale)
    test_acc = bt_accuracy(ratings, test_matchups, test_outcomes, base=base, scale=scale)
    print(f'train acc: {train_acc:.6f}\ttest acc: {test_acc:.6f}')
    mask = test_outcomes != 0.5
    test_acc_no_draw = bt_accuracy(ratings, test_matchups[mask], test_outcomes[mask], base=base, scale=scale)
    print(f'test acc no draw: {test_acc_no_draw:.6f}')

    metrics = {
        'train_nll' : train_nll,
        'test_nll' : test_nll,
        'train_acc' : train_acc,
        'test_acc' :  test_acc,
        'test_acc_no_draw' : test_acc_no_draw
    }
    return metrics, ratings


def rk_eval(
    train_matchups,
    train_outcomes,
    test_matchups,
    test_outcomes,
    get_ratings_fn,
    theta=1.0,
):
    ratings = get_ratings_fn(train_matchups, train_outcomes, theta=theta)
    train_nll = -rk_log_likelihood(ratings, train_matchups, train_outcomes, theta=theta)
    test_nll = -rk_log_likelihood(ratings, test_matchups, test_outcomes, theta=theta)
    print(f'train nll: {train_nll:.6f}\ttest nll: {test_nll:.6f}')
    train_acc = rk_accuracy(ratings, train_matchups, train_outcomes, theta=theta)
    test_acc = rk_accuracy(ratings, test_matchups, test_outcomes, theta=theta)
    print(f'train acc: {train_acc:.6f}\ttest acc: {test_acc:.6f}')
    mask = test_outcomes != 0.5
    test_acc_no_draw = rk_accuracy(ratings, test_matchups[mask], test_outcomes[mask], theta=1.0)
    print(f'test acc no draw: {test_acc_no_draw:.6f}')
    metrics = {
        'train_nll' : train_nll,
        'test_nll' : test_nll,
        'train_acc' : train_acc,
        'test_acc' :  test_acc,
        'test_acc_no_draw' : test_acc_no_draw
    }
    return metrics, ratings


def eval_seed(df, seed=0, verbose=False):
    train_df, test_df = split(df, test_size=0.2, shuffle=True, seed=seed)
    train_matchups, train_outcomes, competitors = preprocess(train_df)
    test_matchups, test_outcomes, _ = preprocess(test_df)
    draw_rate = (train_outcomes == 0.5).mean()
    if verbose: print(f'{draw_rate=}')

    k = 4.0
    base = 10.0
    scale = 400.0
    elo_fn = partial(get_elo_ratings, k=k)
    print(f'evaluating elo: {k=}, {base=}, {scale=}')
    elo_metrics, elo_ratings = bt_eval(train_matchups, train_outcomes, test_matchups, test_outcomes, elo_fn, base, scale)
    elo_metrics['method'] = 'elo'
    if verbose: print_top_k(elo_ratings, competitors)
    print('')

    bootstrap_elo_fn = partial(get_bootstrap_elo_ratings, num_bootstrap=100, k=k)
    print(f'evaluating bootstrap elo: {k=}, {base=}, {scale=}')
    bootstrap_elo_metrics, bootstrap_elo_ratings = bt_eval(train_matchups, train_outcomes, test_matchups, test_outcomes, bootstrap_elo_fn, base, scale)
    bootstrap_elo_metrics['method'] = 'bootstrap elo'
    if verbose: print_top_k(bootstrap_elo_ratings, competitors)
    print('')

    base = math.e
    scale=1.0
    bt_fn = partial(get_bt_ratings_lbfgs, base=base, scale=scale)
    print(f'evaluating lbfgs bt {base=}, {scale=}')
    bt_metrics, bt_ratings = bt_eval(train_matchups, train_outcomes, test_matchups, test_outcomes, bt_fn, base, scale)
    bt_metrics['method'] = 'bt'
    if verbose: print_top_k(bt_ratings, competitors)
    print('')

    theta = 2.0
    max_iter = 10
    ilsr_fn = partial(get_ilsr_ratings, theta=theta, max_iter=max_iter, eps=1e-6)
    print(f'evaluating ilsr rk {theta=}, {max_iter=}')
    ilsr_metrics, ilsr_ratings = rk_eval(train_matchups, train_outcomes, test_matchups, test_outcomes, ilsr_fn, theta=theta)
    ilsr_metrics['method'] = 'ilsr'
    if verbose: print_top_k(ilsr_ratings, competitors)
    print('')

    rk_fn = get_rk_ratings_lbfgs
    print(f'evaluating rk lbfgs {theta=}')
    rk_metrics, rk_ratings = rk_eval(train_matchups, train_outcomes, test_matchups, test_outcomes, rk_fn, theta=theta)
    rk_metrics['method'] = 'rk_lbfgs'
    if verbose: print_top_k(rk_ratings, competitors)
    print('')


    metrics = [elo_metrics, bootstrap_elo_metrics, bt_metrics, ilsr_metrics, rk_metrics]
    for metric in metrics:
        metric['seed'] = seed
    return metrics


if __name__ == '__main__':
    df = pd.read_json('chatbot_arena_conversations_july_.json', lines=True)
    metrics = []
    num_seeds = 100
    for seed in range(num_seeds):
        seed_metrics = eval_seed(df, seed=seed, verbose=False)
        metrics.extend(seed_metrics)
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.groupby(['method']).mean())