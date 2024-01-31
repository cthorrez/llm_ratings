import math
from functools import partial
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.utils.optimize import _newton_cg
from opt import diag_hess_newtons_method


def bt_loss_and_grad(ratings, matchups, outcomes, base=10., s=400., eps=1e-6):
    n_competitors = ratings.shape[0]
    # mask of shape [num_matchups, 2, num_competitors]
    schedule_mask = np.equal(matchups[:, :, None], np.arange(n_competitors)[None,:])
    rating_diff = ratings[matchups[:,1]] - ratings[matchups[:,0]]
    probs = 1.0 / (1.0 + np.power(base, rating_diff / s))
    probs = np.clip(probs, eps, 1 - eps)
    loss_array = -(outcomes * np.log(probs)) - ((1.0 - outcomes) * np.log(1.0 - probs))
    loss = loss_array.mean()
    grad = (outcomes - probs)[:,None]
    grad = np.repeat(grad,  axis=1, repeats=2)
    grad[:,0] *= -1.0
    grad = (grad[:,:,None] * schedule_mask).sum(axis=(0,1)) / matchups.shape[0]
    # don't forget the chain rule!
    grad = grad * (np.log(base) / s)
    return loss, grad

def bt_hess_vec_prod(ratings, vec, matchups, outcomes, base=10., s=400., eps=1e-6):
    n_competitors = ratings.shape[0]
    # mask of shape [num_matchups, 2, num_competitors]
    schedule_mask = np.equal(matchups[:, :, None], np.arange(n_competitors)[None,:])
    rating_diff = ratings[matchups[:,1]] - ratings[matchups[:,0]]
    probs = 1.0 / (1.0 + np.power(base, rating_diff / s))
    probs = np.clip(probs, eps, 1 - eps)
    hess_diag = ((probs * (1.0 - probs))[:,None,None] * schedule_mask).mean(axis=(0,1))
    return hess_diag * vec

def bt_grad_hess(ratings, vec, matchups, outcomes, base=10., s=400., eps=1e-6):
    n_competitors = ratings.shape[0]
    # mask of shape [num_matchups, 2, num_competitors]
    schedule_mask = np.equal(matchups[:, :, None], np.arange(n_competitors)[None,:])
    rating_diff = ratings[matchups[:,1]] - ratings[matchups[:,0]]
    probs = 1.0 / (1.0 + np.power(base, rating_diff / s))
    probs = np.clip(probs, eps, 1 - eps)
    grad = (outcomes - probs)[:,None]
    grad = np.repeat(grad,  axis=1, repeats=2)
    grad[:,0] *= -1.0
    grad = (grad[:,:,None] * schedule_mask).mean(axis=(0,1))
    hess_diag = ((probs * (1.0 - probs))[:,None,None] * schedule_mask).mean(axis=(0,1))
    def hvp(x):
        return hess_diag * x
    return grad, hvp

def bt_f_grad_hess(ratings, matchups, outcomes, base=10., s=400., eps=1e-6):
    n_competitors = ratings.shape[0]
    # mask of shape [num_matchups, 2, num_competitors]
    schedule_mask = np.equal(matchups[:, :, None], np.arange(n_competitors)[None,:])
    rating_diff = ratings[matchups[:,1]] - ratings[matchups[:,0]]
    probs = 1.0 / (1.0 + np.power(base, rating_diff / s))
    probs = np.clip(probs, eps, 1 - eps)
    loss_array = -(outcomes * np.log(probs)) - ((1.0 - outcomes) * np.log(1.0 - probs))
    loss = loss_array.mean()
    grad = (outcomes - probs)[:,None]
    grad = np.repeat(grad,  axis=1, repeats=2)
    grad[:,0] *= -1.0
    grad = (grad[:,:,None] * schedule_mask).mean(axis=(0,1))
    hess_diag = ((probs * (1.0 - probs))[:,None,None] * schedule_mask).mean(axis=(0,1))
    return loss, grad, hess_diag



def calc_probs_bt(matchups, ratings, base, scale):
    alpha = math.log(base) / scale
    all_ratings = ratings[matchups]
    probs = expit(alpha * (all_ratings[:,0] - all_ratings[:,1]))
    return probs


def get_bt_ratings_lbfgs(matchups, outcomes, base=10., scale=400.0):
    num_competitors = np.max(matchups) + 1
    ratings = np.zeros(num_competitors)
    ratings = minimize(
        fun=partial(bt_loss_and_grad, base=base, s=scale),
        x0=ratings,
        args = (matchups, outcomes),
        method='L-BFGS-B',
        jac=True,
        options={'disp' : False}
    )['x']
    return ratings

def get_bt_probs_lbfgs(matchups, outcomes, base=10., scale=400.0):
    ratings = get_bt_ratings_lbfgs(matchups, outcomes, base, scale)
    probs = calc_probs_bt(matchups, ratings, base=base, s=scale)
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
