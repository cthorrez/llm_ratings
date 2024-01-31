from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eval_utils import eval_seed
from data_utils import split, preprocess, generate_matchup_data


if __name__ == '__main__':

    config = {
        'num_matchups': 100000,
        'num_competitors': 200,
        'strength_var': 1.0,
        'strength_noise_var': [0.25, 0.5, 1.0, 1.25],
        'theta' : 1.0,
    }
    x_axis_key = 'strength_noise_var'

    competitor_cols = ['competitor_1', 'competitor_2']
    outcome_col = 'outcome'

    metrics = []
    num_seeds = 10
    y_axis_vals = []
    for x_axis_val in config[x_axis_key]:
        for seed in range(42, num_seeds + 42):
            seed_config = deepcopy(config)
            seed_config['seed'] = seed
            seed_config[x_axis_key] = x_axis_val

            df = generate_matchup_data(**seed_config)
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
        mean_metrics = pd.DataFrame(metrics).groupby(['method']).mean()
        print(mean_metrics['test_acc'])
        y_axis_vals.append(mean_metrics['test_acc'])

    plt.plot(config[x_axis_key], y_axis_vals)
    plt.xlabel(x_axis_key)
    plt.ylabel('test_acc')
    plt.legend(mean_metrics.index)
    plt.show()

    