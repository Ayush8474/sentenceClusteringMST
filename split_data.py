import pandas as pd


if __name__ == "__main__":
    data = pd.read_csv("data_new.csv")

    data_category_range = data['airline_sentiment'].unique()
    data_category_range = data_category_range.tolist()

    for i,value in enumerate(data_category_range):
        data[data['airline_sentiment'] == value].to_csv(r'sentiment_'+str(value)+r'.csv',index = False, na_rep = 'N/A')
