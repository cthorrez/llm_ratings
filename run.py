import time
import pandas as pd
from datasets import load_dataset
from elo import compute_elo, compute_elo_mle, predict_win_rate, pretty_print_elo_ratings

def main():

    dataset = load_dataset("lmsys/chatbot_arena_conversations")
    matches = pd.DataFrame(dataset['train']).sort_values(ascending=True, by=["tstamp"])
    matches_no_ties = matches[~matches["winner"].str.contains("tie")]

    ratings = compute_elo(matches)
    pretty_print_elo_ratings(ratings)

    mle_ratings = compute_elo_mle(matches)
    pretty_print_elo_ratings(mle_ratings)

if __name__ == '__main__':
    main()

