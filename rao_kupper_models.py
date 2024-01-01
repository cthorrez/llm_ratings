import math
from functools import partial
import numpy as np
from scipy.optimize import minimize
import kickscore as ks

def get_rao_kupper_ratings(matchups, outcomes, obs_type="logit", margin=None, theta=None, var=1.0):
    if margin is not None:
        margin = margin
    if theta is not None:
        margin = math.log(theta)

    model = ks.TernaryModel(margin=margin, obs_type=obs_type)
    k = ks.kernel.Constant(var=var)

    num_competitors = np.max(matchups) + 1
    for item in range(num_competitors):
        model.add_item(item, kernel=k)

    for matchup, outcome in zip(matchups, outcomes):
        is_tie = outcome == 0.5
        winner, loser = matchup
        if outcome == 0.0:
            winner, loser = loser, winner
        model.observe(winners=[winner], losers=[loser], t=0.0, tie=is_tie)
    model.fit(verbose=False, tol=1e-6, max_iter=100)

    print(f'model mean neg log likelihood: {-model.log_likelihood / matchups.shape[0]}')

    ratings = []
    for item in range(num_competitors):
        ms, vs = model.item[item].predict([0.0])
        score = ms[0]
        ratings.append(score)

    ratings = np.array(ratings)
    ratings = np.exp(ratings)   # this seems to fit them in log scale so we do this before computing metrics
    return ratings


def get_rao_kupper_probs(matchups, outcomes, obs_type="logit", var=1.0):
    draw_prob = (outcomes == 0.5).mean()
    # draw_margin = math.exp(draw_prob)
    draw_margin = draw_prob / 2
    print(f'{draw_margin=}')

    model = ks.TernaryModel(margin=draw_margin, obs_type=obs_type)
    k = ks.kernel.Constant(var=var)

    for item in np.unique(matchups):
        model.add_item(item, kernel=k)

    for matchup, outcome in zip(matchups, outcomes):
        is_tie = outcome == 0.5
        winner, loser = matchup
        if outcome == 0.0:
            winner, loser = loser, winner
        model.observe(winners=[winner], losers=[loser], t=0.0, tie=is_tie)
    model.fit(verbose=False, tol=1e-6, max_iter=100)

    print(f'model mean neg log likelihood: {-model.log_likelihood / matchups.shape[0]}')

    probs = np.empty_like(outcomes)
    for idx, (comp_1, comp_2) in enumerate(matchups):
        prob, _, _ = model.probabilities([comp_1], [comp_2], t=0.0, margin=0.0)
        probs[idx] = prob
    return probs


def get_rk_ratings_lbfgs(matchups, outcomes, theta=1.0):
    num_competitors = np.max(matchups) + 1
    ratings = np.zeros(num_competitors)
    ratings = minimize(
        fun=partial(rk_loss_and_grad, theta=theta),
        x0=ratings,
        args = (matchups, outcomes),
        method='L-BFGS-B',
        jac=True,
        options={'disp' : False}
    )['x']
    ratings = np.exp(ratings)
    return ratings

def calc_probs_rk(ratings, matchups, theta=1.0):
    pi = ratings
    pi_1 = pi[matchups[:,0]]
    pi_2 = pi[matchups[:,1]]
    denom_1 = pi_1 + (theta * pi_2)
    denom_2 = pi_2 + (theta * pi_1)
    prob_1_win = pi_1 / denom_1
    prob_2_win = pi_2 / denom_2
    num = pi_1 * pi_2 * (np.square(theta) - 1.0)
    prob_draw = num / (denom_1 * denom_2)
    return prob_1_win, prob_draw, prob_2_win

def rk_loss_and_grad(ratings, matchups, outcomes, theta, eps=1e-6):
    pi = ratings
    pi = np.exp(ratings)
    pi_1 = pi[matchups[:,0]]
    pi_2 = pi[matchups[:,1]]
    n_competitors = ratings.shape[0]
    # mask of shape [num_matchups, 2, num_competitors]
    schedule_mask = np.equal(matchups[:, :, None], np.arange(n_competitors)[None,:])

    prob_1_win, prob_draw, prob_2_win = calc_probs_rk(
        pi,
        matchups,
        theta=theta
    )
    win_1_mask = outcomes == 1.0
    win_2_mask = outcomes == 0.0
    draw_mask = outcomes == 0.5
    win_1_loglike = np.log(prob_1_win[win_1_mask])
    win_2_loglike = np.log(prob_2_win[win_2_mask])
    draw_loglike = np.log(prob_draw[draw_mask])
    outcome_loglike = np.concatenate([win_1_loglike, win_2_loglike, draw_loglike])
    loss = -outcome_loglike.mean()

    dlp1win_dp1 = (1.0 / pi_1) - (1.0 / (pi_1 + (theta * pi_2)))
    dlp1win_dp2 = - theta / (pi_1 + (theta * pi_2))
    dlp2win_dp1 = - theta / (pi_2 + (theta * pi_1))
    dlp2win_dp2 = (1.0 / pi_2) - (1.0 / (pi_2 + (theta * (pi_1))))
    dldraw_dp1 = dlp1win_dp1 + dlp2win_dp1
    dldraw_dp2 = dlp1win_dp2 + dlp2win_dp2
       
    grad = np.zeros(shape=matchups.shape, dtype=np.float64)
    grad[win_1_mask,0] = dlp1win_dp1[win_1_mask]
    grad[win_2_mask,0] = dlp2win_dp1[win_2_mask]
    grad[draw_mask,0] = dldraw_dp1[draw_mask]
    grad[win_1_mask,1] = dlp1win_dp2[win_1_mask]
    grad[win_2_mask,1] = dlp2win_dp2[win_2_mask]
    grad[draw_mask,1] = dldraw_dp2[draw_mask]

    # do negative sign since it's the NEGATIVE log likelihood
    grad *= -1
    grad = (grad[:,:,None] * schedule_mask).mean(axis=(0,1))
    
    # grad = np.exp(ratings) * grad
    return loss, grad

   