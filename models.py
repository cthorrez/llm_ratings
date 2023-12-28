import math
import numpy as np
from scipy import optimize
from rao_kupper_models import calc_probs_rk

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
    grad = (grad[:,:,None] * schedule_mask).mean(axis=(0,1))
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

def bt_nll(ratings, vec, matchups, outcomes, base=10., s=400., eps=1e-6):
    n_competitors = ratings.shape[0]
    # mask of shape [num_matchups, 2, num_competitors]
    schedule_mask = np.equal(matchups[:, :, None], np.arange(n_competitors)[None,:])
    rating_diff = ratings[matchups[:,1]] - ratings[matchups[:,0]]
    probs = 1.0 / (1.0 + np.power(base, rating_diff / s))
    probs = np.clip(probs, eps, 1 - eps)
    loss_array = -(outcomes * np.log(probs)) - ((1.0 - outcomes) * np.log(1.0 - probs))
    loss = loss_array.mean()
    return loss

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
