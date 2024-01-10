import os
import gdown
from datasets import load_dataset
import pandas as pd

def preprocess_july_data():
    dataset = load_dataset("lmsys/chatbot_arena_conversations")
    df = pd.DataFrame(dataset['train'])
    df = df[df['anony'] == True]
    df = df[['tstamp', 'model_a', 'model_b', 'winner']]
    df = df.sort_values(ascending=True, by=["tstamp"])
    df['outcome'] = df['winner'].map({'model_a': 1.0, 'model_b': 0.0}).fillna(0.5)
    df.to_json('chatbot_arena_conversations_july.json', orient='records', lines=True)
    

def preprocess_december_data():
    # url from https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH?usp=sharing
    url = "https://drive.google.com/file/d/1vv_-tI_hGIJSraHUGRz_9P8ip3qUCD9E/view"
    filename = gdown.download(url, quiet=False, fuzzy=True)
    df = pd.read_json(filename)
    df = df[df['anony'] == True].sort_values(ascending=True, by=["tstamp"])
    df = df[['tstamp', 'model_a', 'model_b', 'winner']]
    df['outcome'] = df['winner'].map({'model_a': 1.0, 'model_b': 0.0}).fillna(0.5)
    df.to_json('chatbot_arena_conversations_december.json', orient='records', lines=True)
    os.remove(filename)



if __name__ == '__main__':
    preprocess_july_data()
    preprocess_december_data()