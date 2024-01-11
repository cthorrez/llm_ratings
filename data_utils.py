import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from riix.utils.data_utils import RatingDataset


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

def preprocess(df):
    dataset = RatingDataset(
        df=df,
        competitor_cols=['model_a', 'model_b'],
        outcome_col='outcome',
        timestamp_col='tstamp',
        verbose=True,
    )
    matchups = dataset.matchups
    outcomes = dataset.outcomes
    competitors = dataset.idx_to_competitor
    return matchups, outcomes, competitors

def print_top_k(ratings, competitors, k=100):
    k = min(k, ratings.shape[0])
    sorted_idxs = np.argsort(-ratings)
    for print_idx in range(k):
        comp_idx = sorted_idxs[print_idx]
        comp = competitors[comp_idx]
        rating = ratings[comp_idx]
        print(f'{comp:<30s}{rating:.6f}')
    