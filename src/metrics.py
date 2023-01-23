import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import contextily
from sys import argv

'''
This program takes in the edited listings and reviews files,
processes the reviews for each listings according to the review date
(contained within the reviews.csv file in ISO format) and produces
the plots/maps for the metrics defined in my interim report.
'''

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

def total_reviews_per_month(df):
    reviews_count = df.groupby(['tract_code', 'year', 'month']).size().rename('monthly_count')
    reviews_count = reviews_count.reset_index()
    reviews_count = reviews_count.pivot_table(index=['tract_code','month'], columns='year', values='monthly_count')
    reviews_count = reviews_count.fillna(0) # replace NaN with 0 for months that don't have any reviews

    # Plot a line graph for each unique tract_code
    for code in df['tract_code'].unique():
        plt.figure()
        #errors = reviews_count.loc[code].std(axis=1) # axis=1 means get std() using columns (across years for each month)
        tract_df = reviews_count.loc[code]
        #tract_df.plot(title=f'Tract = {code}')
        plt.xlabel('Time (Months)')
        plt.ylabel('Monthly Review Count')
        for year in tract_df.columns:
            plt.errorbar(tract_df.index, tract_df[year], yerr=tract_df[year].sem(), capsize=5, capthick=2, label=year, ecolor='gray')
        
        plt.title(f'Monthly Count for Tract Code {code}')
        plt.legend(title="Years")
        #plt.show()
        #plt.savefig(f'{code}.png', format='png', bbox_inches='tight')
        plt.close()


def total_reviews_per_year(df):
    # Count the number of reviews per year for each unique tract code
    review_count = df.groupby(['tract_code', 'year'])['index'].count()
    review_count = review_count.fillna(0) # replace NaN with 0 for years that don't have reviews

    # Plot the review count per year for each unique tract code
    for code in df['tract_code'].unique():
        plt.figure()
        tract_df = review_count.loc[code]
        plt.xlabel('Time (Years)')
        plt.ylabel('Yearly Review Count')
        plt.errorbar(tract_df.index, tract_df, yerr=tract_df.sem(), fmt='-o', capsize=5, capthick=2)
        plt.xlim((2017.5, 2022.5))
        plt.ylim(bottom=0)
        plt.title(f'Review Count for Tract Code {code}')
        #plt.show()
        #plt.savefig(f'{code}.svg', format='svg', bbox_inches='tight')
        plt.close()
    

def cumulative_reviews_per_month(df):
    # Create a new dataframe with a month-year column
    df['month_year'] = df['year'].astype(str) + '-' + df['month'].astype(str)

    # Count the cumulative amount of reviews per month-year for each unique tract code
    review_count = df.groupby(['tract_code', 'month_year']).size().groupby(level=0).cumsum()
    review_count = review_count.fillna(0) # replace NaN with 0 for months that don't have any reviews

    # Plot the cumulative review count per month-year for each unique tract code
    for code in df['tract_code'].unique():
        plt.figure()
        tract_review_count = review_count.loc[code]
        tract_review_count.plot(xlabel='Month-Year', ylabel='Cumulative Review Count', title=f'Monthly Cumulative Reviews for Tract Code {code}')
        #plt.show()
        #plt.savefig(f'{code}.png', format='png', bbox_inches='tight')
        plt.close()

def cumulative_reviews_per_year(df):
    # Group dataframe by 'tract_code' and 'year'
    reviews_by_tract = df.groupby(['tract_code', 'year']).size()

    # Calculate cumulative sum of reviews
    reviews_by_tract = reviews_by_tract.groupby(level=[0]).cumsum()

    # Reshape dataframe so that rows are time period (year) and columns are 'tract_code'
    reviews_by_tract = reviews_by_tract.reset_index()
    reviews_by_tract = reviews_by_tract.pivot_table(index=['year'], columns='tract_code', values=0)
    reviews_by_tract = reviews_by_tract.fillna(0) # replace NaN with 0 for months that don't have any reviews

    # Create line plot for each 'tract_code'
    for tract_code in reviews_by_tract.columns:
        plt.figure()
        reviews_by_tract[tract_code].plot(title=f'Yearly Cumulative Reviews for Tract Code {tract_code}')
        plt.xlabel('Time (year)')
        plt.ylabel('Cumulative Reviews')
        plt.xlim((2017.5, 2022.5))
        plt.ylim(bottom=0)
        #plt.show()
        #plt.savefig(f'{tract_code}.png', format='png', bbox_inches='tight')
        plt.close()

def new_listings_per_month(df):
    pass

def new_listings_per_year(df):
    pass




if __name__ == "__main__":
    reviews_df = separate_dates(argv[1])
    listings_df = pd.read_csv(argv[2])
    #total_reviews_per_month(reviews_df)
    #total_reviews_per_year(reviews_df)
    #cumulative_reviews_per_month(reviews_df)
    cumulative_reviews_per_year(reviews_df)
