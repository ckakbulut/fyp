import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
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


# returns a dataframe where the reviews.csv is organized into each tracts' monthly review count
def city_monthly_total_reviews(df):
    reviews_count = df.groupby(['tract_code', 'year', 'month']).size().rename('monthly_count')
    reviews_count = reviews_count.reset_index()
    reviews_count = reviews_count.pivot_table(index='tract_code', columns = ['year', 'month'], values='monthly_count')
    reviews_count = reviews_count.fillna(0) # replace NaN with 0 for months that don't have any reviews
    print(reviews_count.head())

    # Plot a box chart showing the IQR for each month between tracts 
    months = reviews_count.columns
    years = [str(year) for year, month in months]
    months_array = [str(month) for year, month in months]
    unique_years = sorted(set(years), key=lambda x: int(x))

    fig, ax = plt.subplots(2, figsize=(20, 10))

    # Plot the box plots for each month
    for i, month in enumerate(months):
        ax[0].boxplot(reviews_count[month].values, positions=[i], widths=0.6)
        ax[1].boxplot(reviews_count[month].values, positions=[i], widths=0.6)

    # Set the x-axis labels for the first subplot (years)
    ax[0].set_xticks(range(0, len(months), 12))
    ax[0].set_xticklabels(unique_years)

    # Set the x-axis labels for the second subplot (months)
    ax[1].set_xticks(range(0, len(months)))
    ax[1].set_xticklabels(months_array)

    # Set the y-axis label
    ax[0].set_ylabel("Total Monthly Review Count")
    ax[1].set_ylabel("Total Monthly Review Count")

    # Show the plot
    plt.show()

    return reviews_count

def city_monthly_first_review_count(df):
    # Convert the date column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Group by listing_id and date and keep only the first occurrence of each listing_id
    df = df.groupby('listing_id').first().reset_index()

    # Set the date column as the index of the dataframe
    df = df.set_index('date')

    # Resample the dataframe to count the number of new listings per month
    new_listings_per_month = df.resample('M').size()

    # Slice the dataframe to include only the data between January 2018 and September 2022
    new_listings_per_month = new_listings_per_month['2018-01-01':'2022-09-01']

    # Plot the result
    plt.plot(new_listings_per_month.index, new_listings_per_month.values)
    plt.xlabel("Month")
    plt.ylabel("New Listings")
    plt.title("City New Listings")
    plt.show()


if __name__ == "__main__":
    reviews_df = separate_dates(argv[1])
    listings_df = pd.read_csv(argv[2])
    city_monthly_total_reviews(reviews_df)
    #city_monthly_first_review_count(reviews_df)
