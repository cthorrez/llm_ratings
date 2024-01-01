import math
import numpy as np
from rao_kupper_models import calc_probs_rk


def bt_log_likelihood(ratings, matchups, outcomes, base=math.e, scale=1.0):
    diffs = ratings[matchups[:,0]] - ratings[matchups[:,1]]
    probs = 1.0 / (1.0 + np.power(base, -diffs / scale))
    # treats a draw as half a win and half a loss
    # for a draw outcome=0.5 so its 0.5*log(prob) + 0.5*log(1-prob)
    nll = (outcomes * np.log(probs)) + ((1.0 - outcomes) * np.log(1.0 - probs))
    return nll.mean()

def rk_log_likelihood(ratings, matchups, outcomes, theta=1.0):
    prob_1_win, prob_draw, prob_2_win = calc_probs_rk(
        ratings,
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
    return outcome_loglike.mean()
