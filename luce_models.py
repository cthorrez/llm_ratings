import time
import math
import numpy as np
from scipy.special import expit
import pandas as pd
from datasets import load_dataset
from riix.utils.data_utils import RatingDataset
from riix.eval import evaluate
from riix.metrics import binary_metrics_suite
from luce import wlsr_ties, mm_ties, ilsr_ties, imm_ties

SQRT2 = math.sqrt(2.0)

def preprocess_for_luce(matchups, outcomes):
    draw_mask = outcomes == 0.5
    ties = matchups[draw_mask,:]
    comparisons = matchups[~draw_mask,:]
    swap_mask = outcomes[~draw_mask] == 0.0
    comparisons[swap_mask,:] = comparisons[swap_mask,::-1] # winner has to be first
    return comparisons, ties

def calc_probs_bt(matchups, ratings):
    all_ratings = ratings[matchups]
    probs = expit(all_ratings[:,0] - all_ratings[:,1])
    return probs

def get_ilsr_ratings(matchups, outcomes, theta=SQRT2, max_iter=1000):
    comparisons, ties = preprocess_for_luce(matchups, outcomes)
    num_competitors = np.max(matchups) + 1
    ratings = ilsr_ties(
        n=num_competitors,
        comparisons=comparisons,
        ties=ties,
        theta=theta,
        max_iter=max_iter,
        tol=1e-6
    )
    return ratings


def get_ilsr_probs(matchups, outcomes, theta=SQRT2, max_iter=1000):
    ratings = get_ilsr_ratings(matchups, outcomes, theta, max_iter)
    probs = calc_probs_bt(matchups, ratings)
    return probs


def get_mm_probs(matchups, outcomes, theta=SQRT2, max_iter=1000):
    comparisons, ties = preprocess_for_luce(matchups, outcomes)
    num_competitors = np.max(matchups) + 1

    ratings = imm_ties(
        n=num_competitors,
        comparisons=comparisons,
        ties=ties,
        theta=theta,
        max_iter=max_iter,
        tol=1e-8
    )
    probs = calc_probs_bt(matchups, ratings)
    return probs





def main():
    matches = pd.read_json('clean_battle_anony_20231206.json')
    matches['outcome'] = matches['winner'].map({'model_a': 1.0, 'model_b': 0.0}).fillna(0.5)
    # matches = matches[~matches["winner"].str.contains("tie")].reset_index()


    # matches = matches.head(1000)

    dataset = RatingDataset(
        df=matches,
        competitor_cols=['model_a', 'model_b'],
        outcome_col='outcome',
        timestamp_col='tstamp',
        batch_size=1,
    )
    matchups = dataset.matchups
    outcomes = dataset.outcomes

    comparisons, ties = preprocess_for_luce(matchups, outcomes)
    n = dataset.num_competitors

    wlsr_ratings = wlsr_ties(
        n=n,
        comparisons=comparisons,
        ties=ties
    )
    # idxs = np.argsort(-wlsr_ratings)
    # for idx in range(idxs.shape[0]):
    #     print(f'{idx+1}: {dataset.idx_to_competitor[idxs[idx]]}\t\t{wlsr_ratings[idxs[idx]]}')
    wlsr_probs = calc_probs_bt(matchups, wlsr_ratings)
    wlsr_metrics = binary_metrics_suite(wlsr_probs, outcomes)
    print(f'{wlsr_metrics=}')

    ilsr_ratings = ilsr_ties(
        n=n,
        comparisons=comparisons,
        ties=ties,
        tol=1e-8
    )
    # idxs = np.argsort(-wlsr_ratings)
    # for idx in range(idxs.shape[0]):
    #     print(f'{idx+1}: {dataset.idx_to_competitor[idxs[idx]]}\t\t{wlsr_ratings[idxs[idx]]}')
    ilsr_probs = calc_probs_bt(matchups, ilsr_ratings)
    ilsr_metrics = binary_metrics_suite(ilsr_probs, outcomes)
    print(f'{ilsr_metrics=}')


    mm_ratings = mm_ties(
        n=n,
        comparisons=comparisons,
        ties=ties
    )
    # idxs = np.argsort(-mm_ratings)
    # for idx in range(idxs.shape[0]):
    #     print(f'{idx+1}: {dataset.idx_to_competitor[idxs[idx]]}\t\t{mm_ratings[idxs[idx]]}')
    mm_probs = calc_probs_bt(matchups, mm_ratings)
    mm_metrics = binary_metrics_suite(mm_probs, outcomes)
    print(f'{mm_metrics=}')



if __name__ == '__main__':
    main()