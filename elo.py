"""
Module for Elo ratings on LLM preference data
source from: https://colab.research.google.com/drive/1J2Wf7sxc9SVmGnSX_lImhT246pxNVZip?usp=sharing
"""
import math
import numpy as np
import pandas as pd
from riix.models.elo import Elo


def get_elo_ratings(matchups, outcomes, k, base=math.e, scale=1.0):
    num_competitors = np.max(matchups) + 1
    model = Elo(
        num_competitors=num_competitors,
        initial_rating=0.0,
        k=k,
        alpha=math.log(base) / scale,
        update_method='iterative'
    )
    _ = model.fit(time_step=0, matchups=matchups, outcomes=outcomes)
    return model.ratings

def get_bootstrap_elo_ratings(matchups, outcomes, k, base=math.e, scale=1.0, num_bootstrap=100):
    num_matchups = matchups.shape[0]
    num_competitors = np.max(matchups) + 1
    all_ratings = np.zeros(shape=(num_bootstrap, num_competitors))
    for sample_idx in range(num_bootstrap):
        rng = np.random.default_rng(seed=sample_idx)
        idxs = rng.choice(np.arange(num_matchups), size=num_matchups, replace=True)
        sample_matchups = matchups[idxs]
        sample_outcomes = outcomes[idxs]
        model = Elo(
            num_competitors=num_competitors,
            initial_rating=0.0,
            k=k,
            alpha=math.log(base) / scale,
            update_method='iterative'
        )
        _ = model.fit(time_step=0, matchups=sample_matchups, outcomes=sample_outcomes)
        all_ratings[sample_idx,:] = model.ratings
    return np.mean(all_ratings, axis=0)
