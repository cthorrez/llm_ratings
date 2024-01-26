import time
import math
import numpy as np
from scipy.special import expit
import pandas as pd
from datasets import load_dataset
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

def get_ilsr_ratings(matchups, outcomes, theta=SQRT2, max_iter=1000, eps=1e-6, do_log_transform=False):
    comparisons, ties = preprocess_for_luce(matchups, outcomes)
    num_competitors = np.max(matchups) + 1
    ratings = ilsr_ties(
        n=num_competitors,
        comparisons=comparisons,
        ties=ties,
        theta=theta,
        max_iter=max_iter,
        tol=eps,
        do_log_transform=do_log_transform
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
