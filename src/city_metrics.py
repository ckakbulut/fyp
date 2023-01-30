import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from sys import argv


def separate_dates(reviews):
    df = pd.read_csv(reviews)
    # sort reviews by date
    df = df.sort_values(by=['date'], ascending=True)
    # separate date of reviews into year, month and day for easy access to metrics
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    return df




if __name__ == "__main__":
    #reviews_df = separate_dates(argv[1])
    #istings_df = pd.read_csv(argv[2])
    #total_reviews_per_month(reviews_df)
    #total_reviews_per_year(reviews_df)
    #cumulative_reviews_per_month(reviews_df)
    #cumulative_reviews_per_year(reviews_df)
