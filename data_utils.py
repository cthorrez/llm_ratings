import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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
    outcomes = df[outcome_col].values.astype(np.float64)
    print(f'num matchups: {matchups.shape[0]}')
    print(f'num competitors: {num_competitors}')
    return matchups, outcomes, competitors

def print_top_k(ratings, competitors, k=100):
    k = min(k, ratings.shape[0])
    sorted_idxs = np.argsort(-ratings)
    for print_idx in range(k):
        comp_idx = sorted_idxs[print_idx]
        comp = competitors[comp_idx]
        rating = ratings[comp_idx]
        print(f'{comp:<30s}{rating:.6f}')
    