import gdown
import pandas as pd

def main():
    # url from https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH?usp=sharing
    url = "https://drive.google.com/file/d/1vv_-tI_hGIJSraHUGRz_9P8ip3qUCD9E/view"
    filename = gdown.download(url, quiet=False, fuzzy=True)
    df = pd.read_json(filename).sort_values(ascending=True, by=["tstamp"])
    df = df[df['anony'] == True]
    print(df.shape)
    print(df.columns)

if __name__ == '__main__':
    main()