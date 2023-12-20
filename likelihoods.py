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
    pass