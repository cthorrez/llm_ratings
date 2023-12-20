import pandas as pd

def main():

    matches = pd.read_json('clean_battle_anony_20231206.json')
    matches['outcome'] = matches['winner'].map({'model_a': 1.0, 'model_b': 0.0}).fillna(0.5)
    # matches = matches[~matches["winner"].str.contains("tie")].reset_index()
    print(len(matches))

    cond_1 = (matches['model_a'] == 'gpt-4-turbo') & (matches['model_b'] == 'gpt-4-0314')
    cond_2 = (matches['model_b'] == 'gpt-4-turbo') & (matches['model_a'] == 'gpt-4-0314')
    df = matches[cond_1 | cond_2].reset_index()
    print(len(df))

    cond_1 = (df['model_a'] == 'gpt-4-turbo') & (df['model_b'] == 'gpt-4-0314')
    cond_2 = (df['model_a'] == 'gpt-4-0314') & (df['model_b'] == 'gpt-4-turbo')

    win_0314 = len(df[cond_1 & (df['outcome'] == 0.0)]) + len(df[cond_2 & (df['outcome'] == 1.0)])

    win_turbo = len(df[cond_1 & (df['outcome'] == 1.0)]) + len(df[cond_2 & (df['outcome'] == 0.0)])

    ties = len(df[df['outcome'] == 0.5])

    print(f'{win_0314=}')
    print(f'{win_turbo=}')
    print(f'{ties=}')

if __name__ ==  '__main__':
    main()