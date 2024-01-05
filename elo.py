"""
Module for Elo ratings on LLM preference data
source from: https://colab.research.google.com/drive/1J2Wf7sxc9SVmGnSX_lImhT246pxNVZip?usp=sharing
"""
import math
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from riix.models.elo import Elo


def compute_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)

    for _, model_a, model_b, winner in battles[['model_a', 'model_b', 'winner']].itertuples():
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == "model_a":
            sa = 1
        elif winner == "model_b":
            sa = 0
        elif winner == "tie" or winner == "tie (bothbad)":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)

    return rating

def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {
        a: [wins[a][b] if a != b else np.NAN for b in names]
        for a in names
    }

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T, wins

def predict_probs(df, pred_win_rates, SCALE=400, BASE=10, INIT_RATING=1000):
    probs = np.zeros(len(df))
    for idx, model_a, model_b in df[['model_a', 'model_b']].itertuples():
        prob = pred_win_rates[model_a][model_b]
        probs[idx] = prob
    return probs

def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    from sklearn.linear_model import LogisticRegression
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx)//2:] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X,Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    return pd.Series(elo_scores, index = models.index).sort_values(ascending=False)

def pretty_print_elo_ratings(ratings):
    df = pd.DataFrame([
        [n, ratings[n]] for n in ratings.keys()
    ], columns=["Model", "Elo rating"]).sort_values("Elo rating", ascending=False).reset_index(drop=True)
    df["Elo rating"] = (df["Elo rating"] + 0.5).astype(int)
    df.index = df.index + 1
    print(df)


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
