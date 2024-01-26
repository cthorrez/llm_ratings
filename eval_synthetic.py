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


def eval_seed(df, seed=0, verbose=False, competitor_cols=['model_a', 'model_b'], outcome_col=['outcome']):
    train_df, test_df = split(df, test_size=0.2, shuffle=True, seed=seed)
    train_matchups, train_outcomes, competitors = preprocess(train_df, competitor_cols, outcome_col)
    test_matchups, test_outcomes, _ = preprocess(test_df, competitor_cols, outcome_col)
    draw_rate = (train_outcomes == 0.5).mean()
    if verbose: print(f'{draw_rate=}')
    metrics = []

    k = 4.0
    base = 10.0
    scale = 400.0
    # elo_fn = partial(get_elo_ratings, k=k)
    # print(f'evaluating elo: {k=}, {base=}, {scale=}')
    # elo_metrics, elo_ratings = evaluate(train_matchups, train_outcomes, test_matchups, test_outcomes, elo_fn, 'bt', base=base, scale=scale)
    # elo_metrics['method'] = 'elo'
    # metrics.append(elo_metrics)
    # if verbose: print_top_k(elo_ratings, competitors)
    # print('')

    num_boot = 100
    bootstrap_elo_fn = partial(get_bootstrap_elo_ratings, num_bootstrap=num_boot, k=k, seed=seed)
    print(f'evaluating bootstrap elo: {k=}, {base=}, {scale=}')
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

    theta = 2.0
    max_iter = 10
    ilsr_fn = partial(get_ilsr_ratings, theta=theta, max_iter=max_iter, eps=1e-6)
    print(f'evaluating ilsr rk {theta=}, {max_iter=}')
    ilsr_metrics, ilsr_ratings = evaluate(train_matchups, train_outcomes, test_matchups, test_outcomes, ilsr_fn, 'rk', theta=theta)
    ilsr_metrics['method'] = 'ilsr'
    metrics.append(ilsr_metrics)
    if verbose: print_top_k(ilsr_ratings, competitors)
    print('')

    # rk_fn = get_rk_ratings_lbfgs
    # print(f'evaluating rk lbfgs {theta=}')
    # rk_metrics, rk_ratings = evaluate(train_matchups, train_outcomes, test_matchups, test_outcomes, rk_fn, 'rk', theta=theta)
    # rk_metrics['method'] = 'rk_lbfgs'
    # metrics.append(rk_metrics)
    # if verbose: print_top_k(rk_ratings, competitors)
    # print('')

    for metric in metrics:
        metric['seed'] = seed
    return metrics


if __name__ == '__main__':
    seed = 0
    competitor_cols = ['competitor_1', 'competitor_2']
    outcome_col = 'outcome'
    df = generate_matchup_data(
        num_matchups=100000,
        num_competitors=100,
        num_rating_periods=100,
        skill_var=1.0,
        outcome_noise_var=1.0,
        draw_margin=0.0,
        seed=seed
    )

    metrics = []
    num_seeds = 20
    for seed in range(num_seeds):
        seed_metrics = eval_seed(
            df,
            seed=seed,
            verbose=False,
            competitor_cols=competitor_cols,
            outcome_col=outcome_col
        )
        metrics.extend(seed_metrics)
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.groupby(['method']).mean())