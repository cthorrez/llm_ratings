import math
import numpy as np
import kickscore as ks


def get_rao_kupper_ratings(matchups, outcomes, obs_type="logit", margin=0.1, var=1.0):
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

    return np.array(ratings)


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


def calc_probs_rk(ratings, matchups, outcomes, theta=1.0):
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
   