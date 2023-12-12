import time
import pandas as pd
from datasets import load_dataset
from riix.models.elo import Elo
from riix.models.glicko import Glicko
from riix.utils.data_utils import RatingDataset
from riix.eval import evaluate

def main():
    chat_dataset = load_dataset("lmsys/chatbot_arena_conversations")
    matches = pd.DataFrame(chat_dataset['train']).sort_values(ascending=True, by=["tstamp"])
    matches['outcome'] = matches['winner'].map({'model_a': 1.0, 'model_b': 0.0}).fillna(0.5)
    matches = matches[~matches["winner"].str.contains("tie")].reset_index()

    dataset = RatingDataset(
        df=matches,
        competitor_cols=['model_a', 'model_b'],
        outcome_col='outcome',
        timestamp_col='tstamp',
        batch_size=1,
    )

    elo_model = Elo(
        num_competitors=dataset.num_competitors,
        update_method='batched'
    )
    elo_metrics = evaluate(elo_model, dataset)
    print(elo_metrics)

    glicko_model = Glicko(
        num_competitors=dataset.num_competitors,
        update_method='batched',
        c = 0.0 # no reason to increase variance over time for static models
    )
    glicko_metrics = evaluate(glicko_model, dataset)
    print(glicko_metrics)




    



if __name__ == '__main__':
    main()