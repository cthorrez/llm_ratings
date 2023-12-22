import math
import pandas as pd
import numpy as np
from riix.utils.data_utils import RatingDataset
from riix.metrics import binary_metrics_suite

from luce_models import get_ilsr_ratings, get_ilsr_probs, get_mm_probs
from bradley_terry_models import get_bt_probs_lbfgs, get_bt_probs_newtoncg
from rao_kupper import get_rao_kupper_ratings, get_rao_kupper_probs
from likelihoods import bt_log_likelihood, rk_log_likelihood

def main():
    matches = pd.read_json('clean_battle_anony_20231206.json')
    matches['outcome'] = matches['winner'].map({'model_a': 1.0, 'model_b': 0.0}).fillna(0.5)
    # matches = matches[~matches["winner"].str.contains("tie")].reset_index()

    draw_prob = matches["winner"].str.contains("tie").mean()
    print(draw_prob)

    dataset = RatingDataset(
        df=matches,
        competitor_cols=['model_a', 'model_b'],
        outcome_col='outcome',
        timestamp_col='tstamp',
        batch_size=1,
    )
    matchups = dataset.matchups
    outcomes = dataset.outcomes
    theta = math.sqrt(2.0)
    # theta = 1.0
    # theta = 1.2
    # theta = math.exp(draw_prob)
    print(f'{theta=}')
    margin = math.log(theta)
    print(f'{margin=}')

    ilsr_ratings = get_ilsr_ratings(matchups, outcomes, theta=theta, max_iter=100)
    ilsr_bt = bt_log_likelihood(ilsr_ratings, matchups, outcomes)
    ilsr_rk = rk_log_likelihood(ilsr_ratings, matchups, outcomes, theta=theta)
    print(f'{ilsr_bt=}')
    print(f'{ilsr_rk=}')


    wslr_probs = get_ilsr_probs(matchups, outcomes, theta=theta, max_iter=1)
    wslr_metrics = binary_metrics_suite(wslr_probs, outcomes)
    print(f'{wslr_metrics=}')

    # islr_probs = get_ilsr_probs(matchups, outcomes, theta=theta, max_iter=100)
    # islr_metrics = binary_metrics_suite(islr_probs, outcomes)
    # print(f'{islr_metrics=}')

    # mm_probs = get_mm_probs(matchups, outcomes, theta=theta, max_iter=1)
    # mm_metrics = binary_metrics_suite(mm_probs, outcomes)
    # print(f'{mm_metrics=}')

    # imm_probs = get_mm_probs(matchups, outcomes, theta=theta)
    # imm_metrics = binary_metrics_suite(imm_probs, outcomes)
    # print(f'{imm_metrics=}')

    rk_ratings = get_rao_kupper_ratings(matchups, outcomes, margin=margin)
    rk_bt = bt_log_likelihood(rk_ratings, matchups, outcomes)
    rk_rk = rk_log_likelihood(rk_ratings, matchups, outcomes, theta=theta)
    print(f'{rk_bt=}')
    print(f'{rk_rk=}')
    # rao_kupper_probs = get_rao_kupper_probs(matchups, outcomes)
    # rao_kupper_metrics = binary_metrics_suite(rao_kupper_probs, outcomes)
    # print(f'{rao_kupper_metrics=}')

    # lbfgs_probs = get_lbfgs_probs(matchups, outcomes, base=10.0, s=400.0)
    # lbfgs_metrics = binary_metrics_suite(lbfgs_probs, outcomes)
    # print(f'base 10 {lbfgs_metrics=}')

    # lbfgs_probs = get_lbfgs_probs(matchups, outcomes, base=math.e, s=1.0)
    # lbfgs_metrics = binary_metrics_suite(lbfgs_probs, outcomes)
    # print(f'base e {lbfgs_metrics=}')

    # newtoncg_probs = get_newtoncg_probs(matchups, outcomes)
    # newtoncg_metrics = binary_metrics_suite(newtoncg_probs, outcomes)
    # print(f'{newtoncg_metrics=}')


if __name__ == '__main__':
    main()