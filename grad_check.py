import math
import numpy as np
from scipy.special import expit as sigmoid
import jax.numpy as jnp
from jax import grad, jit, value_and_grad

def rk_loss_fn(ratings, matchups, outcomes, theta, eps=1e-6):
    # pi = ratings
    pi = jnp.exp(ratings)
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
    # pi = ratings
    pi = np.exp(ratings)
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
    
    grad_mean = np.exp(ratings) * grad_mean
    return loss, grad_mean


def bt_loss_fn(ratings, matchups, outcomes, base=10., s=400., eps=1e-6):
    n_competitors = ratings.shape[0]
    # mask of shape [num_matchups, 2, num_competitors]
    schedule_mask = jnp.equal(matchups[:, :, None], jnp.arange(n_competitors)[None,:])
    rating_diff = ratings[matchups[:,1]] - ratings[matchups[:,0]]
    probs = 1.0 / (1.0 + jnp.power(base, rating_diff / s))
    probs = jnp.clip(probs, eps, 1 - eps)
    loss_array = -(outcomes * jnp.log(probs)) - ((1.0 - outcomes) * jnp.log(1.0 - probs))
    loss = loss_array.mean()
    return loss


def bt_loss_and_grad(ratings, matchups, outcomes, base=10., s=400., eps=1e-6):
    n_competitors = ratings.shape[0]
    # mask of shape [num_matchups, 2, num_competitors]
    schedule_mask = np.equal(matchups[:, :, None], np.arange(n_competitors)[None,:])
    rating_diff = ratings[matchups[:,0]] - ratings[matchups[:,1]]
    alpha = np.log(base) / s
    probs = sigmoid(alpha * rating_diff)
    probs = np.clip(probs, eps, 1 - eps)
    loss_array = -(outcomes * np.log(probs)) - ((1.0 - outcomes) * np.log(1.0 - probs))
    loss = loss_array.mean()
    grad = (outcomes - probs)[:,None]
    grad = np.repeat(grad,  axis=1, repeats=2)
    grad[:,0] *= -1.0
    grad = (grad[:,:,None] * schedule_mask).sum(axis=(0,1)) / matchups.shape[0]
    grad = grad * (np.log(base) / s)
    return loss, grad


def main():
    ratings = np.array([0.0, 0.1, -0.1], dtype=np.float64)
    matchups = np.array([[0,1],[1,2],[2,0],[1,0],[2,1]])
    outcomes = np.array([1.0, 0.5, 0.0, 1.0,0.5])
    theta = 2.0
    epsilon = 1e-8

    rk_loss, rk_grad = rk_loss_and_grad(ratings, matchups, outcomes, theta, epsilon)
    print('mine')
    print(rk_loss)
    print(rk_grad)

    jax_rk_loss_and_grad = value_and_grad(rk_loss_fn, argnums=[0])
    jax_rk_loss, jax_rk_grad = jax_rk_loss_and_grad(ratings, matchups, outcomes, theta, epsilon)
    print('auto')
    print(jax_rk_loss)
    print(jax_rk_grad[0])

    base = 10.0
    scale = 400.0
    bt_loss, bt_grad = bt_loss_and_grad(ratings, matchups, outcomes, base, scale, epsilon)
    print('mine')
    print(bt_loss)
    print(bt_grad)

    jax_bt_loss_and_grad = value_and_grad(bt_loss_fn, argnums=[0])
    jax_bt_loss, jax_bt_grad = jax_bt_loss_and_grad(ratings, matchups, outcomes, base, scale, epsilon)
    print('auto')
    print(jax_bt_loss)
    print(jax_bt_grad[0])

if __name__ == '__main__':
    main()