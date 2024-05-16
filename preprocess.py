import os
import requests
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
    

def preprocess_google_drive_data(url, output_fname):
    filename = gdown.download(url, quiet=False, fuzzy=True)
    df = pd.read_json(filename)
    df = df[df['anony'] == True].sort_values(ascending=True, by=['tstamp'])
    df = df[['tstamp', 'model_a', 'model_b', 'winner']]
    df['outcome'] = df['winner'].map({'model_a': 1.0, 'model_b': 0.0}).fillna(0.5)
    df.to_json(f'{output_fname}.json', orient='records', lines=True)
    os.remove(filename)

def preprocess_google_storage_data(url, output_fname):
    tmp_file_name = 'tmp_data.json'
    response = requests.get(url)
    with open(tmp_file_name, 'wb') as file:
        file.write(response.content)
    with open(tmp_file_name, 'r') as file:
        df = pd.read_json(file)
    df = df[df['anony'] == True].sort_values(ascending=True, by=['tstamp'])
    df = df[['tstamp', 'model_a', 'model_b', 'winner']]
    df['outcome'] = df['winner'].map({'model_a': 1.0, 'model_b': 0.0}).fillna(0.5)
    df.to_json(f'{output_fname}.json', orient='records', lines=True)
    os.remove(tmp_file_name)


if __name__ == '__main__':
    preprocess_hf_data('lmsys/chatbot_arena_conversations', 'chatbot_arena_hf')                      # 
    google_drive_args = [
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
        # {
        #     'url' : 'https://drive.google.com/file/d/1Kpg6HD1QCrytCVT7FgRvZhY885TnmpEo/view',
        #     'date': '03-15-2024'
        # }
    ]
    google_storage_args = [
        {
            'url' : 'https://storage.googleapis.com/arena_external_data/public/clean_battle_20240508.json',
            'date': '05-08-2024'
        }
    ]
    for args in google_drive_args:
        preprocess_google_drive_data(
            url=args['url'],
            output_fname=f'chatbot_arena_{args["date"]}'
        )
    for args in google_storage_args:
        preprocess_google_storage_data(
            url=args['url'],
            output_fname=f'chatbot_arena_{args["date"]}'
        )
