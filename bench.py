import math
import pandas as pd
import numpy as np
from riix.utils.data_utils import RatingDataset
from riix.metrics import binary_metrics_suite

from run_luce import get_ilsr_probs, get_mm_probs
from run_batched_models import get_lbfgs_probs, get_newtoncg_probs
from rao_kupper import get_rao_kupper_probs

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
    theta = 1.0
    # theta = 1.2
    # theta = math.exp(draw_prob)
    print(theta)

    

    wslr_probs = get_ilsr_probs(matchups, outcomes, theta=theta, max_iter=1)
    wslr_metrics = binary_metrics_suite(wslr_probs, outcomes)
    print(f'{wslr_metrics=}')

    # islr_probs = get_ilsr_probs(matchups, outcomes, theta=theta, max_iter=100)
    # islr_metrics = binary_metrics_suite(islr_probs, outcomes)
    # print(f'{islr_metrics=}')

    mm_probs = get_mm_probs(matchups, outcomes, theta=theta, max_iter=1)
    mm_metrics = binary_metrics_suite(mm_probs, outcomes)
    print(f'{mm_metrics=}')

    # imm_probs = get_mm_probs(matchups, outcomes, theta=theta)
    # imm_metrics = binary_metrics_suite(imm_probs, outcomes)
    # print(f'{imm_metrics=}')

    rao_kupper_probs = get_rao_kupper_probs(matchups, outcomes)
    rao_kupper_metrics = binary_metrics_suite(rao_kupper_probs, outcomes)
    print(f'{rao_kupper_metrics=}')

    lbfgs_probs = get_lbfgs_probs(matchups, outcomes)
    lbfgs_metrics = binary_metrics_suite(lbfgs_probs, outcomes)
    print(f'{lbfgs_metrics=}')

    lbfgs_probs = get_lbfgs_probs(matchups, outcomes, base=math.e, s=1.0)
    lbfgs_metrics = binary_metrics_suite(lbfgs_probs, outcomes)
    print(f'{lbfgs_metrics=}')

    # newtoncg_probs = get_newtoncg_probs(matchups, outcomes)
    # newtoncg_metrics = binary_metrics_suite(newtoncg_probs, outcomes)
    # print(f'{newtoncg_metrics=}')


if __name__ == '__main__':
    main()