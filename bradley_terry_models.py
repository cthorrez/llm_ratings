import math
from functools import partial
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.utils.optimize import _newton_cg
from riix.utils.data_utils import RatingDataset
from riix.metrics import binary_metrics_suite
from models import bt_loss_and_grad, bt_hess_vec_prod, bt_f_grad_hess
from opt import diag_hess_newtons_method

ALPHA = math.log(10.0) / 400.0
ALPHA = 1.0

def calc_probs_bt(matchups, ratings, base, scale):
    alpha = math.log(base) / scale
    all_ratings = ratings[matchups]
    probs = expit(alpha * (all_ratings[:,0] - all_ratings[:,1]))
    return probs


def get_bt_ratings_lbfgs(matchups, outcomes, base=10., s=400.0):
    num_competitors = np.max(matchups) + 1
    ratings = np.zeros(num_competitors)
    ratings = minimize(
        fun=partial(bt_loss_and_grad, base=base, s=s),
        x0=ratings,
        args = (matchups, outcomes),
        method='L-BFGS-B',
        jac=True,
        options={'disp' : False}
    )['x']
    return ratings

def get_bt_probs_lbfgs(matchups, outcomes, base=10., s=400.0):
    ratings = get_bt_ratings_lbfgs(matchups, outcomes, base, s)
    probs = calc_probs_bt(matchups, ratings, base=base, s=s)
    return probs

def get_bt_probs_newtoncg(matchups, outcomes, base=10., s=400.0):
    num_competitors = np.max(matchups) + 1
    ratings = np.zeros(num_competitors)
    ratings = minimize(
        fun=partial(bt_loss_and_grad, base=base, s=s),
        x0=ratings,
        args = (matchups, outcomes),
        method='newton-cg',
        jac=True,
        hessp=partial(bt_hess_vec_prod, base=base, s=s),
        options={'disp' : False}
    )['x']
    probs = calc_probs_bt(matchups, ratings, base=base, s=s)
    return probs


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

    ratings = np.zeros(dataset.num_competitors)
    matchups = dataset.matchups
    outcomes = dataset.outcomes

    fit_ratings = minimize(
        fun=partial(bt_loss_and_grad, base=math.e, s=1.0),
        x0=ratings,
        args = (matchups, outcomes),
        method='L-BFGS-B',
        jac=True,
        hessp=partial(bt_hess_vec_prod, base=math.e, s=1.0),
        options={'disp' : False}
    )
    # print(fit_ratings)
    fit_ratings = fit_ratings['x']

    # fit_ratings = diag_hess_newtons_method(
    #     x0=ratings,
    #     f_grad_hess=bt_f_grad_hess,
    #     args={'matchups' : matchups, 'outcomes': outcomes},
    # )
    # idxs = np.argsort(-fit_ratings)
    # for idx in range(idxs.shape[0]):
    #     print(f'{idx+1}: {dataset.idx_to_competitor[idxs[idx]]}\t\t{fit_ratings[idxs[idx]]}')

    probs = calc_probs_bt(matchups, fit_ratings)
    metrics = binary_metrics_suite(probs, outcomes)
    print(f'{metrics=}')

if __name__ == '__main__':
    main()