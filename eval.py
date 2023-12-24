import math
import numpy as np
from functools import partial
from data_utils import load_and_split, preprocess, print_top_k
from likelihoods import bt_log_likelihood, rk_log_likelihood
from bradley_terry_models import get_bt_ratings_lbfgs
from rao_kupper_models import get_rao_kupper_ratings
from luce_models import get_ilsr_ratings
from elo import get_elo_ratings
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
    return ratings


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
    return ratings


def main(seed=0, verbose=False):
    train_df, test_df = load_and_split(test_size=0.2, shuffle=True, seed=seed)
    train_matchups, train_outcomes, competitors = preprocess(train_df)
    test_matchups, test_outcomes, _ = preprocess(test_df)
    draw_rate = (train_outcomes == 0.5).mean()
    print(f'{draw_rate=}')

    k = 4.0
    base = 10.0
    scale = 400.0
    elo_fn = partial(get_elo_ratings, k=k)
    print(f'evaluating elo: {k=}, {base=}, {scale=}')
    elo_ratings = bt_eval(train_matchups, train_outcomes, test_matchups, test_outcomes, elo_fn, base, scale)
    if verbose: print_top_k(elo_ratings, competitors)
    print('')

    base = math.e
    scale=1.0
    bt_fn = partial(get_bt_ratings_lbfgs, base=base, scale=scale)
    print(f'evaluating lbfgs bt {base=}, {scale=}')
    bt_ratings = bt_eval(train_matchups, train_outcomes, test_matchups, test_outcomes, bt_fn, base, scale)
    if verbose: print_top_k(bt_ratings, competitors)
    print('')

    theta = 2.0
    max_iter = 2
    ilsr_fn = partial(get_ilsr_ratings, theta=theta, max_iter=max_iter, eps=1e-6)
    print(f'evaluating ilsr rk {theta=}, {max_iter=}')
    ilsr_ratings = rk_eval(train_matchups, train_outcomes, test_matchups, test_outcomes, ilsr_fn, theta=theta)
    if verbose: print_top_k(ilsr_ratings, competitors)
    print('')

    # kickscore_fn = partial(get_rao_kupper_ratings, theta=theta)
    # print(f'evaluating kickscore kr {theta=}')
    # rk_eval(train_matchups, train_outcomes, test_matchups, test_outcomes, kickscore_fn, theta=theta)
    # print('')

if __name__ == '__main__':
    for seed in range(10):
        main(seed=seed)