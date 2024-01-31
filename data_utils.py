import time
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_matchup_data(
    num_matchups: int = 1000,
    num_competitors: int = 100,
    num_rating_periods: int = 10,
    strength_var: float = 1.0,
    strength_noise_var: float = 0.1,
    theta: float = 1.0,
    seed: int = 0,
):
    start_time = int(time.time())
    matchups_per_period = num_matchups // num_rating_periods
    period_offsets = np.arange(num_rating_periods) * (3600 * 24)
    initial_timestamps = np.zeros(num_rating_periods, dtype=np.int64) + start_time
    timestamps = (initial_timestamps + period_offsets).repeat(matchups_per_period)

    rng = np.random.default_rng(seed=seed)
    strength_means = rng.normal(loc=0.0, scale=math.sqrt(strength_var), size=num_competitors)
    comp_1 = rng.integers(low=0, high=num_competitors, size=(num_matchups, 1))
    offset = rng.integers(low=1, high=num_competitors, size=(num_matchups, 1))
    comp_2 = np.mod(comp_1 + offset, num_competitors)
    matchups = np.hstack([comp_1, comp_2])
    strengths = strength_means[matchups]

    strength_noise = rng.normal(loc=0.0, scale=math.sqrt(strength_noise_var), size=strengths.shape)
    strengths = strengths + strength_noise

    probs = np.zeros(shape=(num_matchups, 3))  # p(comp_1 win), p(draw), p(comp_2 win)
    probs[:, 0] = strengths[:, 0] / (strengths[:, 0] + (theta * strengths[:, 1]))
    probs[:, 2] = strengths[:, 1] / (strengths[:, 1] + (theta * strengths[:, 0]))
    probs[:, 1] = 1.0 - probs[:, 0] - probs[:, 2]
    outcomes = np.argmax(probs, axis=1) / 2.0  # map 0->0, 1->0.5, 2->1.0

    data = {
        'timestamp': timestamps,
        'competitor_1': matchups[:, 0],
        'competitor_2': matchups[:, 1],
        'outcome': outcomes,
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df['competitor_1'] = 'competitor_' + df['competitor_1'].astype(str)
    df['competitor_2'] = 'competitor_' + df['competitor_2'].astype(str)
    return df

def split(df, test_size=0.2, seed=0, shuffle=False):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=shuffle
    )
    return train_df, test_df

def load_and_split(path, test_size=0.2, seed=0, shuffle=False):
    df = load(path)
    train_df, test_df = split(df, test_size, seed, shuffle)
    return train_df, test_df

def preprocess(df, competitor_cols=['model_a', 'model_b'], outcome_col=['outcome']):
    competitors = sorted(pd.unique(df[competitor_cols].astype(str).values.ravel('K')).tolist())
    num_competitors = len(competitors)
    competitor_to_idx = {comp: idx for idx, comp in enumerate(competitors)}
    matchups = df[competitor_cols].map(lambda comp: competitor_to_idx[str(comp)]).values.astype(np.int64)
    outcomes = df[outcome_col].values.astype(np.float64).flatten()
    return matchups, outcomes, competitors

def print_top_k(ratings, competitors, k=100):
    k = min(k, ratings.shape[0])
    sorted_idxs = np.argsort(-ratings)
    for print_idx in range(k):
        comp_idx = sorted_idxs[print_idx]
        comp = competitors[comp_idx]
        rating = ratings[comp_idx]
        print(f'{comp:<30s}{rating:.6f}')
    