import math
import numpy as np
from bradley_terry_models import calc_probs_bt
from rao_kupper_models import calc_probs_rk

def bt_accuracy(ratings, matchups, outcomes, draw_margin=0.0, base=math.e, scale=1.0):
    probs = calc_probs_bt(
        matchups=matchups,
        ratings=ratings,
        base=base,
        scale=scale
    )
    preds = np.zeros_like(probs) + 0.5
    preds[probs >= 0.5 + draw_margin] = 1.0
    preds[probs < 0.5 - draw_margin] = 0.0
    correct_wins = np.logical_and(preds==1.0, outcomes==1.0).sum()
    correct_losses = np.logical_and(preds==0.0, outcomes==0.0).sum()
    correct_draws = np.logical_and(preds==0.5, outcomes==0.5).sum()
    return (correct_wins + correct_losses + correct_draws) / matchups.shape[0]

def rk_accuracy(ratings, matchups, outcomes, theta=1.0):
    prob_1_win, prob_draw, prob_2_win = calc_probs_rk(
        ratings,
        matchups,
        outcomes,
        theta=theta
    )
    probs = np.hstack([prob_1_win[:,None], prob_draw[:,None], prob_2_win[:,None]])
    preds = 1.0 - (np.argmax(probs, axis=1) / 2.0) # map 0->1.0, 1->0.5, 2->0.0
    print(f'predicted draw rate: {(preds==0.5).mean()}')
    correct_wins = np.logical_and(preds==1.0, outcomes==1.0).sum()
    correct_losses = np.logical_and(preds==0.0, outcomes==0.0).sum()
    correct_draws = np.logical_and(preds==0.5, outcomes==0.5).sum()
    return (correct_wins + correct_losses + correct_draws) / matchups.shape[0]

