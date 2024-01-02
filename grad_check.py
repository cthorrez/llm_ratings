import numpy as np
import jax.numpy as jnp
from jax import grad, jit, value_and_grad



def rk_loss_fn(ratings, matchups, outcomes, theta, eps=1e-6):
    # pi = jnp.exp(ratings)
    pi = ratings
    pi_1 = pi[matchups[:,0]]
    pi_2 = pi[matchups[:,1]]
    denom_1 = pi_1 + (theta * pi_2)
    denom_2 = pi_2 + (theta * pi_1)
    prob_1_win = pi_1 / denom_1
    prob_2_win = pi_2 / denom_2
    num = pi_1 * pi_2 * (jnp.square(theta) - 1.0)
    prob_draw = num / (denom_1 * denom_2)

    win_1_mask = outcomes == 1.0
    win_2_mask = outcomes == 0.0
    draw_mask = outcomes == 0.5
    win_1_loglike = jnp.log(prob_1_win[win_1_mask])
    win_2_loglike = jnp.log(prob_2_win[win_2_mask])
    draw_loglike = jnp.log(prob_draw[draw_mask])
    outcome_loglike = jnp.concatenate([win_1_loglike, win_2_loglike, draw_loglike])
    loss = -outcome_loglike.mean()
    return loss

def rk_loss_and_grad(ratings, matchups, outcomes, theta, eps=1e-6):
    # pi = np.exp(ratings)
    pi = ratings
    pi_1 = pi[matchups[:,0]]
    pi_2 = pi[matchups[:,1]]
    n_competitors = ratings.shape[0]
    # mask of shape [num_matchups, 2, num_competitors]
    schedule_mask = np.equal(matchups[:, :, None], np.arange(n_competitors)[None,:])

    denom_1 = pi_1 + (theta * pi_2)
    denom_2 = pi_2 + (theta * pi_1)
    prob_1_win = pi_1 / denom_1
    prob_2_win = pi_2 / denom_2
    num = pi_1 * pi_2 * (np.square(theta) - 1.0)
    prob_draw = num / (denom_1 * denom_2)

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
       
    grad = np.zeros(shape=matchups.shape, dtype=np.float32)
    grad[win_1_mask,0] = dlp1win_dp1[win_1_mask]
    grad[win_2_mask,0] = dlp2win_dp1[win_2_mask]
    grad[draw_mask,0] = dldraw_dp1[draw_mask]
    grad[win_1_mask,1] = dlp1win_dp2[win_1_mask]
    grad[win_2_mask,1] = dlp2win_dp2[win_2_mask]
    grad[draw_mask,1] = dldraw_dp2[draw_mask]

    # do negative sign since it's the NEGATIVE log likelihood
    grad *= -1
    grad_expanded = grad[:,:,None] * schedule_mask
    grad_sum = grad_expanded.sum(axis=(0,1))
    grad_mean = grad_sum / outcomes.shape[0]
    
    # grad = np.exp(ratings) * grad
    return loss, grad_mean


def main():
    ratings = np.array([0.0, 0.1, -0.1], dtype=np.float32) + 1.0
    matchups = np.array([[0,1],[1,2],[2,0],[1,0],[2,1]])
    outcomes = np.array([1.0, 0.5, 0.0, 1.0,0.5])
    theta = 2.0
    epsilon = 1e-6

    rk_loss, rk_grad = rk_loss_and_grad(ratings, matchups, outcomes, theta, epsilon)
    print(rk_loss)
    print(rk_grad)

    jax_rk_loss_and_grad = value_and_grad(rk_loss_fn, argnums=[0])
    jax_rk_loss, jax_rk_grad = jax_rk_loss_and_grad(ratings, matchups, outcomes, theta, epsilon)
    print(jax_rk_loss)
    print(jax_rk_grad[0])

if __name__ == '__main__':
    main()