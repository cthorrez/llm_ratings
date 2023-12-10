import time
import pandas as pd
from sklearn.model_selection import train_test_split
from riix.metrics import binary_log_loss as log_loss, binary_accuracy as acc
from datasets import load_dataset
from elo import compute_elo, compute_elo_mle, predict_win_rate, predict_probs, pretty_print_elo_ratings

def main():
    dataset = load_dataset("lmsys/chatbot_arena_conversations")
    matches = pd.DataFrame(dataset['train']).sort_values(ascending=True, by=["tstamp"])
    matches['label'] = matches['winner'].map({'model_a': 1.0, 'model_b': 0.0}).fillna(0.5)
    train_matches, test_matches = train_test_split(
        matches,
        test_size=0.2,
        shuffle=True,
        random_state=1
    )
    train_matches_no_ties = train_matches[~train_matches["winner"].str.contains("tie")].reset_index()
    test_matches_no_ties = test_matches[~test_matches["winner"].str.contains("tie")].reset_index()

    train_matches, test_matches = train_matches_no_ties, test_matches_no_ties

    ratings = compute_elo(train_matches)
    _, pred_win_rates = predict_win_rate(ratings)
    probs = predict_probs(test_matches, pred_win_rates)
    print(f'elo acc: {acc(probs, test_matches["label"].values)}')
    print(f'elo log loss: {log_loss(probs, test_matches["label"].values)}')

    mle_ratings = compute_elo_mle(train_matches)
    _, pred_win_rates_mle = predict_win_rate(mle_ratings)
    mle_probs = predict_probs(test_matches, pred_win_rates_mle)
    print(f'elo mle acc: {acc(mle_probs, test_matches["label"].values)}')
    print(f'elo mle log loss: {log_loss(mle_probs, test_matches["label"].values)}')

if __name__ == '__main__':
    main()

