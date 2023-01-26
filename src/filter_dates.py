import pandas as pd
from sys import argv

'''
This program takes in the listings_detailed.csv file and removes any listings with a last review date before 01-01-2018. Likewise, it takes in the reviews.csv file and removes any reviews left before 01-01-2018.

As output it produces two files, edited_listings.csv and edited_reviews.csv, which are then used as input for other files to process the data.
'''

def filter_listing_dates(listings):
    df = pd.read_csv(listings)
    df = df[df['last_review'] > ('2018-01-01')]
    df.to_csv('filtered_listings.csv', encoding='utf-8')

def filter_review_dates(reviews):
    df = pd.read_csv(reviews)
    df = df[df['date'] > ('2018-01-01')]
    df.to_csv('filtered_reviews.csv', encoding='utf-8')

if __name__ == "__main__":
    filter_listing_dates(argv[1])
    filter_review_dates(argv[2])
