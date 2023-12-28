import math
import numpy as np
from scipy import optimize

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



def rk_loss_and_grad(ratings, matchups, outcomes, theta, eps=1e-6):
    pi = ratings
    pi_1 = pi[matchups[:,0]]
    pi_2 = pi[matchups[:,1]]
    n_competitors = ratings.shape[0]
    # mask of shape [num_matchups, 2, num_competitors]
    schedule_mask = np.equal(matchups[:, :, None], np.arange(n_competitors)[None,:])

    prob_1_win, prob_draw, prob_2_win = calc_probs_rk(
        ratings,
        matchups,
        outcomes,
        theta=theta
    )
    win_1_mask = outcomes == 1.0
    win_2_mask = outcomes == 0.0
    draw_mask = outcomes == 0.5
    win_1_loglike = np.log(prob_1_win[win_1_mask])
    win_2_loglike = np.log(prob_2_win[win_2_mask])
    draw_loglike = np.log(prob_draw[draw_mask])
    outcome_loglike = np.concatenate([win_1_loglike, win_2_loglike, draw_loglike])
    loss = outcome_loglike.mean()

    dlp1win_dp1 = (1.0 / pi_1) - (1.0 / (pi_1 + (theta * pi_2)))
    dlp1win_dp2 = - theta / (pi_1 + (theta * pi_2))
    dlp2win_dp1 = - theta / (pi_2 + (theta * pi_1))
    dlp2win_dp2 = (1.0 / pi_2) - (1.0 / (pi_2 + (theta (pi_1))))
    dldraw_dp1 = dlp1win_dp1 + dlp2win_dp1
    dldraw_dp2 = dlp1win_dp2 + dlp2win_dp2
       
    grad = np.zeros(shape=matchups.shape, dtype=np.float64)
    grad[win_1_mask,0] = dlp1win_dp1[win_1_mask]
    grad[win_2_mask,0] = dlp2win_dp1[win_2_mask]
    grad[draw_mask,0] = dldraw_dp1[draw_mask]
    grad[win_1_mask,1] = dlp1win_dp2[win_1_mask]
    grad[win_2_mask,1] = dlp2win_dp2[win_2_mask]
    grad[draw_mask,1] = dldraw_dp2[draw_mask]

    grad = (grad[:,:,None] * schedule_mask).mean(axis=(0,1))
    return loss, grad