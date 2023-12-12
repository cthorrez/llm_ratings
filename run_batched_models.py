import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.utils.optimize import _newton_cg
from riix.utils.data_utils import RatingDataset
from models import bt_loss_and_grad, bt_hess_vec_prod, bt_f_grad_hess
from opt import diag_hess_newtons_method

def main():
    matches = pd.read_json('clean_battle_anony_20231206.json')
    matches['outcome'] = matches['winner'].map({'model_a': 1.0, 'model_b': 0.0}).fillna(0.5)
    # matches = matches[~matches["winner"].str.contains("tie")].reset_index()
    
    dataset = RatingDataset(
        df=matches,
        competitor_cols=['model_a', 'model_b'],
        outcome_col='outcome',
        timestamp_col='tstamp',
    )

    ratings = np.zeros(dataset.num_competitors) + 1000.0
    matchups = dataset.matchups
    outcomes = dataset.outcomes

    fit_ratings = minimize(
        fun=bt_loss_and_grad,
        x0=ratings,
        args = (matchups, outcomes),
        method='L-BFGS-B',
        jac=True,
        hessp=bt_hess_vec_prod,
        options={'disp' : True}
    )
    print(fit_ratings)
    fit_ratings = fit_ratings['x']

    # fit_ratings = diag_hess_newtons_method(
    #     x0=ratings,
    #     f_grad_hess=bt_f_grad_hess,
    #     args={'matchups' : matchups, 'outcomes': outcomes},
    # )

    idxs = np.argsort(-fit_ratings)
    for idx in range(idxs.shape[0]):
        print(f'{idx+1}: {dataset.idx_to_competitor[idxs[idx]]}\t\t{fit_ratings[idxs[idx]]}')

if __name__ == '__main__':
    main()