"""
Module for Elo ratings on LLM preference data
source from: https://colab.research.google.com/drive/1J2Wf7sxc9SVmGnSX_lImhT246pxNVZip?usp=sharing
"""
import math
import numpy as np
import pandas as pd
from riix.models.elo import Elo
from scipy.special import expit as sigmoid

class VectorizedElo():
    """Fit multiple Elo models at the same time for bootstrapping ;)"""
    def __init__(self, num_competitors, num_models, k=4.0, base=10.0, scale=400.0):
        self.ratings = np.zeros(shape=(num_models, num_competitors), dtype=np.float64)
        self.k = k
        self.alpha = math.log(base) / scale
        self.num_models = num_models

    def fit(self, matchups, outcomes):
        """
            matchups: (num_models, num_matchups, 2)
            outcomes: (num_models, num_matchups)
        """
        model_idxs = np.arange(self.num_models)
        for m_idx in range(matchups.shape[1]):
            c1, c2 = np.split(matchups[:,m_idx,:], 2, axis=1)
            c1 = c1[:,0]
            c2 = c2[:,0]
            r1 = self.ratings[model_idxs,c1]
            r2 = self.ratings[model_idxs,c2]
            probs = sigmoid(self.alpha*(r1 - r2))
            updates = self.k * (outcomes[:,m_idx] - probs)
            self.ratings[model_idxs,c1] += updates
            self.ratings[model_idxs,c2] -= updates
        return np.mean(self.ratings, axis=0)


def get_bootstrap_elo_ratings_vec(matchups, outcomes, k, base=10.0, scale=400.0, num_bootstrap=100):
    rng = np.random.default_rng(seed=42)
    num_matchups = matchups.shape[0]
    idxs = rng.choice(np.arange(num_matchups), size=(num_bootstrap,num_matchups), replace=True)
    boot_matchups = matchups[idxs]
    boot_outcomes = outcomes[idxs]
    num_competitors = np.unique(matchups).shape[0]
    model = VectorizedElo(
        num_competitors=num_competitors,
        num_models=num_bootstrap,
        k=k,
        base=base,
        scale=scale,
    )
    ratings = model.fit(boot_matchups, boot_outcomes)
    return ratings


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
