import pandas as pd
from sys import argv

''' take as input the reviews.csv file for a city, don't filter dates on listings.csv as we want to see the dates for new listings as well'''

def filter_dates(reviews):
    df = pd.read_csv(reviews)
    df = df[df['date'] > ('2018-01-01')]
    df.to_csv('filtered_reviews.csv', encoding='utf-8')

if  __name__ == "__main__":
    for i in range (1, len(argv)):
        filter_dates(argv[i])

    
