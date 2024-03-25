import os
import gdown
from datasets import load_dataset
import pandas as pd

def preprocess_hf_data(hf_dataset, output_fname):
    dataset = load_dataset(hf_dataset)
    df = pd.DataFrame(dataset['train'])
    df = df[df['anony'] == True]
    df = df[['tstamp', 'model_a', 'model_b', 'winner']]
    df = df.sort_values(ascending=True, by=['tstamp'])
    df['outcome'] = df['winner'].map({'model_a': 1.0, 'model_b': 0.0}).fillna(0.5)
    df.to_json(f'{output_fname}.json', orient='records', lines=True)
    

def preprocess_google_data(url, output_fname):
    filename = gdown.download(url, quiet=False, fuzzy=True)
    df = pd.read_json(filename)
    df = df[df['anony'] == True].sort_values(ascending=True, by=['tstamp'])
    df = df[['tstamp', 'model_a', 'model_b', 'winner']]
    df['outcome'] = df['winner'].map({'model_a': 1.0, 'model_b': 0.0}).fillna(0.5)
    df.to_json(f'{output_fname}.json', orient='records', lines=True)
    os.remove(filename)


if __name__ == '__main__':
    preprocess_hf_data('lmsys/chatbot_arena_conversations', 'chatbot_arena_hf')                      # 
    google_args = [
        # {
        #     'url' : 'https://drive.google.com/file/d/1vv_-tI_hGIJSraHUGRz_9P8ip3qUCD9E/view',
        #     'date' : '12-06-2023'
        # },
        # {
        #     'url' : 'https://drive.google.com/file/d/1jjJ8k3L-BzFKSevoGo6yaJ-jCjc2SCK1/view', 
        #     'date' : '01-06-2024'
        # },
        # {
        #     'url' : 'https://drive.google.com/file/d/1O3GGotY8I5d4xgxU-qQyizgKFjpSt4ta/view',
        #     'date' : '01-26-2024'
        # },
        # {
        #     'url' : 'https://drive.google.com/uc?id=1ZXiBRtADf9HZ8eEarIFTy-qrPNDUP_H0',
        #     'date' : '02-15-2024'
        # },
        {
            'url' : 'https://drive.google.com/file/d/1Kpg6HD1QCrytCVT7FgRvZhY885TnmpEo/view',
            'date': '03-15-2024'
        }
    ]
    for args in google_args:
        preprocess_google_data(
            url=args['url'],
            output_fname=f'chatbot_arena_{args["date"]}'
        )
