import math
import numpy as np
from data_utils import load_and_split, preprocess
from likelihoods import bt_log_likelihood, rk_log_likelihood
from bradley_terry_models import get_bt_ratings_lbfgs
from rao_kupper_models import get_rao_kupper_ratings
from luce_models import get_ilsr_ratings
from metrics import bt_accuracy, rk_accuracy

def main():
    train_df, test_df = load_and_split(test_size=0.2, shuffle=True)
    train_matchups, train_outcomes = preprocess(train_df)
    test_matchups, test_outcomes = preprocess(test_df)
    draw_rate = (train_outcomes == 0.5).mean()
    print(f'{draw_rate=}')

    # draw_margin = 0.05
    # theta = math.exp(draw_margin)
    
    theta = math.sqrt(2.0)
    theta = 1.25
    draw_margin = np.log(theta)

    # bt_ratings = get_bt_ratings_lbfgs(train_matchups, train_outcomes, base=math.e, s=1.0)
    # bt_train_nll = -bt_log_likelihood(bt_ratings, train_matchups, train_outcomes)
    # bt_test_nll = -bt_log_likelihood(bt_ratings, test_matchups, test_outcomes)
    # print(f'bt train nll: {bt_train_nll:.6f}\tbt test nll: {bt_train_nll:.4f}')
    # bt_train_acc = bt_accuracy(bt_ratings, train_matchups, train_outcomes, draw_margin=0.0)
    # bt_test_acc = bt_accuracy(bt_ratings, test_matchups, test_outcomes, draw_margin=0.0)
    # print(f'bt train acc: {bt_train_acc:.6f}\tbt test acc: {bt_train_acc:.4f}')

    # rk_ratings = get_rao_kupper_ratings(train_matchups, train_outcomes, margin=theta)
    # rk_train_nll = -rk_log_likelihood(rk_ratings, train_matchups, train_outcomes, theta=theta)
    # rk_test_nll = -rk_log_likelihood(rk_ratings, test_matchups, test_outcomes, theta=theta)
    # print(f'rk train nll: {rk_train_nll:.6f}\trk test nll: {rk_train_nll:.4f}')
    # rk_train_acc = rk_accuracy(rk_ratings, train_matchups, train_outcomes, theta=theta)
    # rk_test_acc = rk_accuracy(rk_ratings, test_matchups, test_outcomes, theta=theta)
    # print(f'rk train acc: {rk_train_acc:.6f}\trk test acc: {rk_train_acc:.4f}')

    ilsr_ratings = get_ilsr_ratings(train_matchups, train_outcomes, theta=theta, max_iter=1)
    ilsr_train_nll = -rk_log_likelihood(ilsr_ratings, train_matchups, train_outcomes, theta=theta)
    ilsr_test_nll = -rk_log_likelihood(ilsr_ratings, test_matchups, test_outcomes, theta=theta)
    print(f'ilsr train nll: {ilsr_train_nll:.6f}\tilsr test nll: {ilsr_train_nll:.4f}')
    ilsr_train_acc = rk_accuracy(ilsr_ratings, train_matchups, train_outcomes, theta=theta)
    ilsr_test_acc = rk_accuracy(ilsr_ratings, test_matchups, test_outcomes, theta=theta)
    print(f'ilsr train acc: {ilsr_train_acc:.6f}\tilsr test acc: {ilsr_train_acc:.4f}')













if __name__ == '__main__':
    main()