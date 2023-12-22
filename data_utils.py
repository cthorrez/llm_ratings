import pandas as pd
from sklearn.model_selection import train_test_split
from riix.utils.data_utils import RatingDataset

def load_and_split(path='clean_battle_anony_20231206.json', test_size=0.2, seed=0, shuffle=False):
    df = matches = pd.read_json(path)
    matches['outcome'] = matches['winner'].map({'model_a': 1.0, 'model_b': 0.0}).fillna(0.5)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=shuffle
    )
    return train_df, test_df

def preprocess(df):
    dataset = RatingDataset(
        df=df,
        competitor_cols=['model_a', 'model_b'],
        outcome_col='outcome',
        timestamp_col='tstamp',
    )
    matchups = dataset.matchups
    outcomes = dataset.outcomes
    return matchups, outcomes
    