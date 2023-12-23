import math
import numpy as np
from functools import partial
from data_utils import load_and_split, preprocess
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
    scale=1.0
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
    test_acc_no_draw = rk_accuracy(ratings, test_matchups[mask], test_outcomes[mask], theta=theta)
    print(f'test acc no draw: {test_acc_no_draw:.6f}')









def main():
    train_df, test_df = load_and_split(test_size=0.2, shuffle=True, seed=0)
    train_matchups, train_outcomes = preprocess(train_df)
    test_matchups, test_outcomes = preprocess(test_df)
    draw_rate = (train_outcomes == 0.5).mean()
    print(f'{draw_rate=}')

    # draw_margin = 0.05
    # theta = math.exp(draw_margin)
    
    theta = math.sqrt(2.0)
    theta = 1.25
    draw_margin = np.log(theta)

    k = 4.0
    base = 10.0
    scale = 400.0
    elo_fn = partial(get_elo_ratings, k=k)
    print(f'evaluating elo: {k=}, {base=}, {scale=}')
    bt_eval(train_matchups, train_outcomes, test_matchups, test_outcomes, elo_fn, base, scale)
    print('')

    base = math.e
    scale=1.0
    bt_fn = partial(get_bt_ratings_lbfgs, base=base, scale=scale)
    print(f'evaluating lbfgs bt {base=}, {scale=}')
    bt_eval(train_matchups, train_outcomes, test_matchups, test_outcomes, bt_fn, base, scale)
    print('')


    theta = 1.1
    ilsr_fn = partial(get_ilsr_ratings, theta=theta, eps=1e-6, do_log_transform=False)
    print(f'evaluating ilsr rk {theta=}')
    rk_eval(train_matchups, train_outcomes, test_matchups, test_outcomes, ilsr_fn, theta=theta)
    print('')

    kickscore_fn = partial(get_rao_kupper_ratings, theta=theta)
    print(f'evaluating kickscore kr {theta=}')
    rk_eval(train_matchups, train_outcomes, test_matchups, test_outcomes, kickscore_fn, theta=theta)
    print('')
    exit(1)

if __name__ == '__main__':
    main()