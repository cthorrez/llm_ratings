import numpy as np
import pandas as pd
from data_utils import split, generate_matchup_data
from eval_utils import eval_seed


if __name__ == '__main__':
    seed = 0
    competitor_cols = ['competitor_1', 'competitor_2']
    outcome_col = 'outcome'


    metrics = []
    num_seeds = 10
    for seed in range(num_seeds):
        df = generate_matchup_data(
            num_matchups=10000,
            num_competitors=50,
            num_rating_periods=100,
            strength_var=1.0,
            strength_noise_var=1.0,
            theta=1.0,
            seed=seed
        )
        train_df, test_df = split(df, test_size=0.2, shuffle=True, seed=seed)
        seed_metrics = eval_seed(
            train_df,
            test_df,
            seed=seed,
            verbose=False,
            competitor_cols=competitor_cols,
            outcome_col=outcome_col
        )
        metrics.extend(seed_metrics)
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.groupby(['method']).mean())