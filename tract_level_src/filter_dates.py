import pandas as pd
from sys import argv

def filter_dates(reviews):
    df = pd.read_csv(reviews)
    df = df[df['date'] > ('2018-01-01')]
    df.to_csv('edited_reviews.csv', encoding='utf-8')

if  __name__ == "__main__":
    for i in range (1, len(argv)):
        filter_dates(argv[i])

    
