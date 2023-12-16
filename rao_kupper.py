import math
import numpy as np
import kickscore as ks


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

    probs = np.empty_like(outcomes)
    for idx, (comp_1, comp_2) in enumerate(matchups):
        prob, _, _ = model.probabilities([comp_1], [comp_2], t=0.0, margin=0.0)
        probs[idx] = prob
    return probs
   