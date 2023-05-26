import pandas as pd

# read CSV file


if __name__ == "__main__":

    df = pd.read_csv('data_new.csv')
    x = df.loc[df['tweet_id'] == 570300616901320704]
    print(x.iloc[0]['text'])