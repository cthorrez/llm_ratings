import math
import numpy as np


def bt_log_likelihood(ratings, matchups, outcomes, base=math.e, scale=1.0):
    diffs = ratings[matchups[:,0]] - ratings[matchups[:,1]]
    probs = 1.0 / (1.0 + np.power(base, diffs / scale))
    # treats a draw as half a win and half a loss
    # for a draw outcome=0.5 so its 0.5*log(prob) + 0.5*log(1-prob)
    nll = (outcomes * np.log(probs)) + ((1.0 - outcomes) * np.log(1.0 - probs))
    return nll.mean()

def rk_log_likelihood(ratings, matchups, outcomes, theta=1.0, base=math.e, scale=1.0):
    pi = np.exp(ratings)
    pi_1 = pi[matchups[:,0]]
    pi_2 = pi[matchups[:,1]]
    denom_1 = pi_1 + (theta * pi_2)
    denom_2 = pi_2 + (theta * pi_1)
    prob_1_win = pi_1 / denom_1
    prob_2_win = pi_2 / denom_2
    num = pi_1 * pi_2 * (np.square(theta) - 1.0)
    prob_draw = num / (denom_1 * denom_2)
    win_1_mask = outcomes == 1.0
    win_2_mask = outcomes == 0.0
    draw_mask = outcomes == 0.5
    win_1_loglike = np.log(prob_1_win[win_1_mask]).sum()
    win_2_loglike = np.log(prob_2_win[win_2_mask]).sum()
    draw_loglike = np.log(prob_draw[draw_mask].sum())
    loglike = (win_1_loglike + win_2_loglike + draw_loglike) / matchups.shape[0]
    return loglike
