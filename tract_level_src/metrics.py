import pandas as pd 
import matplotlib.pyplot as plt
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from datetime import datetime
import re
import census_data

'''
This program takes in the edited listings and reviews files,
processes the reviews for each listings according to the review date
(contained within the reviews.csv file in ISO format) and produces
the dataframes and csv files for the required metrics.

The file must be run from the command line with the following arguments:
    1. The merged and filtered reviews.csv file (containing the tract code associated with each review) -> used to calculate cumulative # reviews per month
    2. The merged and unfiltered reviews.csv file (containing the tract code associated with each review) -> used to calculate cumulative # listings per month
    3. The merged listings.csv file (containing the tract code associated with each listing) -> used to calculate cumulative # new hosts per month
    4. The name of the city which the reviews are in
    5. The 5 digit FIPS code of the state and county which the city is in (used for extracting census data)
'''

def separate_dates(reviews):
    # read in the merged_reviews.csv file and convert the tract code column to a string in order to prevent leading zeros from being deleted
    df = pd.read_csv(reviews, dtype={'tract_code': str})
    # drop any rows that don't have an assigned tract code to the review
    df = df.dropna(subset=['tract_code'])
    # sort reviews by date
    df = df.sort_values(by=['date'], ascending=True)
    # separate date of reviews into year, month and day for easy access to metrics
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    return df


'''calculate the absolute number of reviews each month for a given city
which will be used to plot histogram to show trends in Airbnb usage over time '''
def count_absolute_reviews_per_month(start_date, end_date, df):
    # convert start_date and end_date to datetime objects
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)

    df = df.sort_values(by=['tract_code'])

    # create a new dataframe with date as the index and absolute count as the column
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    counts_df = pd.DataFrame(index=date_range, columns=['count'])

    # iterate through each month and count the number of reviews
    for month_start in date_range:
        month_end = month_start + pd.offsets.MonthEnd(0) # get the last day of the month
        count = len(df[(df['date'] >= month_start) & (df['date'] <= month_end)])
        counts_df.loc[month_start, 'count'] = count

    print(counts_df.head(100))

    # convert the column to an int in order to be able to plot it in a histogram
    counts_df['count'] = counts_df['count'].astype(int)

    counts_df.index = counts_df.index.strftime('%Y-%m')

    # plot a histogram on the cumulative number of reviews per month for the entire city
    ax = counts_df.plot(kind='bar', figsize=(15,10))
    plt.title('Absolute Number of Reviews per Month for the Entire City')
    ax.set_xticklabels('')
    ax.set_xticks([i+0.5 for i in range(len(counts_df.index))], minor=True)
    ax.set_xticklabels([i for i in counts_df.index], rotation=45, fontsize=6, minor=True)
    plt.show()


def count_cumulative_reviews_per_month(start_date, end_date, df):
    # convert start_date and end_date to datetime objects
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)

    df = df.sort_values(by=['tract_code'])

    # create a new dataframe with date as the index and cumulative count as the column
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    counts_df_city = pd.DataFrame(index=date_range, columns=['cumulative_count'])

    # iterate through each month and count the cumulative number of reviews (FOR THE ENTIRE CITY / ALL TRACTS)
    for month_start in date_range:
        month_end = month_start + pd.offsets.MonthEnd(0)
        cum_count = len(df[(df['date'] >= start_date) & (df['date'] <= month_end)])
        counts_df_city.loc[month_start, 'cumulative_count'] = cum_count

    counts_df_city.index = counts_df_city.index.strftime('%Y-%m')
    
    # plot a histogram on the cumulative number of reviews per month for the entire city
    ax = counts_df_city.plot(kind='bar', figsize=(15,10))
    plt.title('Cumulative Number of Reviews per Month for the Entire City')
    ax.set_xticklabels([i for i in counts_df_city.index], rotation=45, fontsize=6)
    plt.show()
    plt.savefig("cumreviews.pdf", format="pdf", bbox_inches="tight")

    # create a new dataframe to store the slope for each tract
    slopes_df = pd.DataFrame(columns=['tract_code', 'slope'])

    # iterate through each tract and count the cumulative number of over the given time period
    for tract in df['tract_code'].unique():
        tract_df = df[df['tract_code'] == tract]

        # create a new dataframe with date as the index and cumulative count as the column
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        counts_df = pd.DataFrame(index=date_range, columns=['cumulative_count'])

        # iterate through each month and count the cumulative number of reviews
        for month_start in date_range:
            month_end = month_start + pd.offsets.MonthEnd(0)
            cum_count = len(tract_df[(tract_df['date'] >= start_date) & (tract_df['date'] <= month_end)])
            counts_df.loc[month_start, 'cumulative_count'] = cum_count

        # convert the dates to a time series in order to be able to use them in the linear regression
        counts_df.index = pd.to_datetime(counts_df.index)
        counts_df.index = [i for i in range(1,len(counts_df.index)+1)]

        # fit a linear regression to the counts_df
        X = counts_df.index.values.reshape(-1, 1)
        y = counts_df['cumulative_count'].values.reshape(-1,1)
        lr = LinearRegression().fit(X, y)

        # store the slope in the slopes_df
        slopes_df = slopes_df.append({'tract_code': tract, 'slope': lr.coef_[0][0]}, ignore_index=True)

    # return the slopes_df
    return slopes_df


def count_new_hosts_per_month(start_date, end_date, df):  
    # sort the dataframe by host_since date for better visualizations during debugging and verification of data
    df = df.sort_values(by=['host_since'], ascending=True)
    
    #convert the start_date and end_date to datetime objects
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)

    # convert the host_since column to a datetime object in order to be able to compare it to the start_date and end_date
    df['host_since'] = pd.to_datetime(df['host_since'])

    # Filter the DataFrame to include only listings within the specified period
    city_df = df[(df['host_since'] >= start_date) & (df['host_since'] <= end_date)]

    # Group the DataFrame by tract_code and host_since, and count the number of unique host_id values
    host_counts = city_df.groupby(pd.Grouper(key='host_since', freq='MS'))['host_id'].nunique()

    # Reset the index to turn the groupby results into a DataFrame
    host_counts = host_counts.reset_index()

    # Rename the host_id column to reflect that it contains counts of hosts
    host_counts = host_counts.rename(columns={'host_id': 'host_count', 'host_since': 'date'})

    # create a new column to store the cumulative number of hosts
    host_counts['cumulative_host_count'] = host_counts['host_count'].cumsum()

    # store the number of total hosts entering the market in the given time period (entire city)
    total_hosts = host_counts['host_count'].sum()
    print(f'Total_hosts: {total_hosts}')

    # create a new dataframe to store the slope for each tract
    slopes_df = pd.DataFrame(columns=['tract_code', 'hosts_slope'])

    # iterate through each tract and count the number of new hosts added each month between the given time interval
    for tract in df['tract_code'].unique():
        tract_df = df[df['tract_code'] == tract]
        
        # Filter the DataFrame to include only listings within the specified period
        tract_df = tract_df[(tract_df['host_since'] >= start_date) & (tract_df['host_since'] <= end_date)]

        # Group the DataFrame by tract_code and host_since, and count the number of unique host_id values
        tract_host_counts = tract_df.groupby(pd.Grouper(key='host_since', freq='MS'))['host_id'].nunique()

        # Reset the index to turn the groupby results into a DataFrame
        tract_host_counts = tract_host_counts.reset_index()

        # Rename the host_id column to reflect that it contains counts of hosts
        tract_host_counts = tract_host_counts.rename(columns={'host_id': 'host_count', 'host_since': 'date'})

        tract_host_counts['cumulative_host_count'] = tract_host_counts['host_count'].cumsum()

        tract_host_counts['tract_code'] = tract
        
        # fill in any missing values with 0
        tract_host_counts = tract_host_counts.fillna(0)

        # convert the dates to a time series in order to be able to use them in the linear regression
        tract_host_counts['date'] = pd.to_datetime(tract_host_counts['date'])
        tract_host_counts.index = [i for i in range(1,len(tract_host_counts['date'])+1)]

        # fit a linear regression to the counts_df
        X = tract_host_counts['date'].values.reshape(-1, 1)
        y = tract_host_counts['cumulative_host_count'].values.reshape(-1,1)
        lr = LinearRegression().fit(X, y)
        
        # store the slope in the slopes_df
        slopes_df = slopes_df.append({'tract_code': tract, 'hosts_slope': lr.coef_[0][0]}, ignore_index=True)
    
    # return the slopes_df
    return slopes_df


def count_new_listings_per_month(start_date, end_date, df):
    ''' Uses the merged_reviews.csv to check for each listing_id when the date of the first review was for a given listing id, and then counts the number of new listings added each month between the given time interval'''

    # create a new dataframe with date as the index and cumulative count as the column
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    df = pd.read_csv(df, dtype={'tract_code' : str})

    # keep only the first instance of each review, which corresponds to the date the listing was added
    city_df = df.sort_values('date').drop_duplicates('listing_id')

    # rename the date column to 'first_review_date'
    city_df = city_df.rename(columns={'date': 'first_review_date'})

    # create a new column 'month_added' that is the month of the first review
    city_df['month_added'] = pd.to_datetime(city_df['first_review_date']).dt.to_period('M')

    # filter the dataframe to include only listings within the specified period
    city_df = city_df[(city_df['first_review_date'] >= start_date) & (city_df['first_review_date'] <= end_date)]

    # group by tract_code and month_added, then count the number of unique listings
    city_month_counts = city_df.groupby(['month_added']).agg({'listing_id': 'nunique'}).reset_index()

    # rename the listing_id column to 'new_listings'
    city_month_counts = city_month_counts.rename(columns={'listing_id': 'new_listings'})

    # keep track of the cumulative number of new listings
    city_month_counts['cumulative_new_listings'] = city_month_counts['new_listings'].cumsum()

    # create a new dataframe to store the slope for each tract
    slopes_df = pd.DataFrame(columns=['tract_code', 'slope'])

    for tract in df['tract_code'].unique():
        tract_df = df[df['tract_code'] == tract]

        # drop duplicate listings and keep only the row with the earliest date (corresponding to the date the listings was added)
        tract_df = tract_df.sort_values('date').drop_duplicates('listing_id')

        # rename the date column to 'first_review_date'
        tract_df = tract_df.rename(columns={'date': 'first_review_date'})

        # create a new column 'month_added' that is the month of the first review
        tract_df['month_added'] = pd.to_datetime(tract_df['first_review_date']).dt.to_period('M')

        # filter the dataframe to include only listings within the specified period
        tract_df = tract_df[(tract_df['first_review_date'] >= start_date) & (tract_df['first_review_date'] <= end_date)]

        # group by tract_code and month_added, then count the number of unique listings
        tract_month_counts = tract_df.groupby(['month_added']).agg({'listing_id': 'nunique'}).reset_index()

        # rename the listing_id column to 'new_listings'
        tract_month_counts = tract_month_counts.rename(columns={'listing_id': 'new_listings'})

        # keep track of the cumulative number of new listings
        tract_month_counts['cumulative_new_listings'] = tract_month_counts['new_listings'].cumsum()


        ''' Although we already calculate the cumulative number of new listings above, the tract_month_counts dataframe misses months inbetween where there are no new listings added. 
        We need to calculate the cumulative number of listings for EACH month (even if there is no new listings added!) in the given time period in order to be able to use the linear regression to calculate the slope.'''

        counts_df = pd.DataFrame(index=date_range, columns=['cumulative_new_listings'])

        # iterate through each month and count the cumulative number of reviews
        for month_start in date_range:
            month_end = month_start + pd.offsets.MonthEnd(0)
            month_end = month_end.strftime('%Y-%m-%d')
            cum_count = len(tract_df[(tract_df['first_review_date'] >= start_date) & (tract_df['first_review_date'] <= month_end)])
            counts_df.loc[month_start, 'cumulative_new_listings'] = cum_count

        # convert the dates to a time series in order to be able to use them in the linear regression
        counts_df.index = pd.to_datetime(counts_df.index)
        counts_df.index = [i for i in range(1,len(counts_df.index)+1)]

        # fit a linear regression to the counts_df
        X = counts_df.index.values.reshape(-1, 1)
        y = counts_df['cumulative_new_listings'].values.reshape(-1,1)
        lr = LinearRegression().fit(X, y)

        # store the slope in the slopes_df
        slopes_df = slopes_df.append({'tract_code': tract, 'slope': lr.coef_[0][0]}, ignore_index=True)

    return slopes_df


def plot_standardized_normal_distribution(df):
    ''' 
    takes in as input the slopes_df calculated from the count_cumulative_reviews_per_month function and returns a normally distributed version of the slopes inside the slopes_df
    '''

    transformations = [
    ("original", lambda x: x),
    ("square root", np.sqrt),
    ("log", np.log1p)
    ]

    columns = df.columns

    for col in columns:
        # extract the column of interest from the dataframe
        if col == 'tract_code':
            continue

        # store the original column values to refer to in the future
        column = df[col]

        #Check if the data is already normally distributed using the Shapiro-Wilk test
        _, p_value = shapiro(df[col])
        
        # Find the transformation that results in the closest Gaussian distribution
        best_transform = None
        best_p_value = 0
        for name, func in transformations:
            transformed_data = df[col].transform(func)
            _, p_value = shapiro(transformed_data)
            if p_value > best_p_value:
                best_transform = (name, transformed_data)
                best_p_value = p_value
        
        # Update the column with the best transformation
        df[col] = best_transform[1]
        print(f"{col}: {best_transform[0]}")
        
        transformed_df = column.transform([np.log1p, np.square, np.sqrt])
        transformed_df['untransformed'] = column

        # # move untransformed column to the front of the dataframe
        # first_column = transformed_df.pop('untransformed')
        # transformed_df.insert(0, 'untransformed', first_column)
        # transformed_df.hist(bins = 20, figsize=(10,10), layout=(3,2), edgecolor='black')
        # plt.suptitle(f'Histograms of Transformed Data for {col}', size=16)
        # plt.show()

        if col != 'slope':
            # Standardize the data
            df[col] = (df[col] - np.mean(df[col])) / np.std(df[col])
    
    return df


def merge_dataframes(df_a, df_b, new_column_name):
    '''function used to create the final dataframe containing the census data for each tract the slope (the metric we defined) for each tract on each row'''


    # Convert the index of df_b to type str
    df_b.index.name = 'tract_code'

    # Merge the two dataframes based on the index of df_a and column names of df_b_t
    # we merge on right because the incoming census data contains all tracts
    # we can then use this to check which tracts contain information and which must be dropped
    # ALTERNATIVE: Merge on "inner" to keep only those tracts that we have reviews for AND have census data for (not all tracts have airbnb reviews in them)
    new_df = df_a.merge(df_b, how='right', left_on='tract_code', right_on='tract_code')

    #Rename the column with the merged values to new_column_name
    new_df.rename(columns={0: f'{new_column_name}'}, inplace=True)

    # convert the new column to type float
    new_df[new_column_name] = new_df[new_column_name].apply(lambda x: re.sub(r'[^0-9\.]', '', str(x)) if x else x)
    new_df[new_column_name] = new_df[new_column_name].apply(lambda x: float(x) if x else x)

    return new_df


''' Description: Read in the census data and merge it with the reviews and listings dataframes, which contain the metrics we will be analyzing

    Parameters: city_name -> The name of the city we are analyzing to be used in the census data file paths
                county_code -> The county code of the city we are analyzing (used for pre-processing of census data)
                reviews_df -> The dataframe containing the reviews metric for the city we are analyzing
                listings_df -> The dataframe containing the listings metric for the city we are analyzing

    Return: reviews_df -> The reviews dataframe with the census data merged into it
            listings_df -> The listings dataframe with the census data merged into it
'''
def read_in_and_merge_census_data(city_name, county_code, reviews_df, listings_df):
    # Read in the census data
    median_property_data = census_data.single_row_data(f'census_data/{city_name}_census_data/{city_name}_median_property.csv', f'{county_code}')
    median_income_data = census_data.single_row_data(f'census_data/{city_name}_census_data/{city_name}_median_income.csv', f'{county_code}')
    income_ineq_data = census_data.single_row_data(f'census_data/{city_name}_census_data/{city_name}_income_ineq.csv', f'{county_code}')
    median_age_data = census_data.median_age_data(f'census_data/{city_name}_census_data/{city_name}_median_age.csv', f'{county_code}')
    age_data = census_data.age_data(f'census_data/{city_name}_census_data/{city_name}_age.csv', f'{county_code}')
    education_data = census_data.educational_attainment_data(f'census_data/{city_name}_census_data/{city_name}_education.csv', f'{county_code}')
    poverty_data = census_data.percentage_poverty_data(f'census_data/{city_name}_census_data/{city_name}_poverty.csv', f'{county_code}')
    unemployment_data = census_data.unemployment_rate_data(f'census_data/{city_name}_census_data/{city_name}_unemployment.csv', f'{county_code}')
    race_data = census_data.race_diversity_data(f'census_data/{city_name}_census_data/{city_name}_race.csv', f'{county_code}')

    # Merge the reviews_df with the census data dataframes
    reviews_df = merge_dataframes(reviews_df, median_property_data, 'median_property_value')
    reviews_df = merge_dataframes(reviews_df, median_income_data, 'median_income')
    reviews_df = merge_dataframes(reviews_df, income_ineq_data, 'income_ineq')
    reviews_df = merge_dataframes(reviews_df, median_age_data, 'median_age')
    reviews_df = merge_dataframes(reviews_df, age_data, 'young_percentage')
    reviews_df = merge_dataframes(reviews_df, education_data, 'education')
    reviews_df = merge_dataframes(reviews_df, poverty_data, 'poverty_percentage')
    reviews_df = merge_dataframes(reviews_df, unemployment_data, 'unemployment')
    reviews_df = merge_dataframes(reviews_df, race_data, 'race_index')

    # Merge the listings_df with the census data dataframes
    listings_df = merge_dataframes(listings_df, median_property_data, 'median_property_value')
    listings_df = merge_dataframes(listings_df, median_income_data, 'median_income')
    listings_df = merge_dataframes(listings_df, income_ineq_data, 'income_ineq')
    listings_df = merge_dataframes(listings_df, median_age_data, 'median_age')
    listings_df = merge_dataframes(listings_df, age_data, 'young_percentage')
    listings_df = merge_dataframes(listings_df, education_data, 'education')
    listings_df = merge_dataframes(listings_df, poverty_data, 'poverty_percentage')
    listings_df = merge_dataframes(listings_df, unemployment_data, 'unemployment')
    listings_df = merge_dataframes(listings_df, race_data, 'race_index')

    return reviews_df, listings_df


''' Description: Convert the columns in the reviews and listings dataframes to numeric values, remove any rows with NaN values or values of 0 (as these will cause issues with the analysis), and remove any rows with a slope value of 0.

    Parameters: reviews_df -> The dataframe containing the reviews metric for the city we are analyzing
                listings_df -> The dataframe containing the listings metric for the city we are analyzing
    
    Return: reviews_df -> The reviews dataframe with the columns converted to numeric values and NaN and 0 values removed
            listings_df -> The listings dataframe with the columns converted to numeric values and NaN and 0 values removed
            reviews_nan_df -> The reviews dataframe with only the NaN values (i.e. the rows that were dropped)
            listings_nan_df -> The listings dataframe with only the NaN values (i.e. the rows that were dropped)
'''
def cleanse_data(reviews_df, listings_df):
    # Convert the columns to numeric values
    reviews_df[['slope', 'median_income', 'median_property_value', 'income_ineq', 'median_age', 'young_percentage', 'education', 'poverty_percentage', 'unemployment', 'race_index']] = reviews_df[['slope', 'median_property_value', 'median_income', 'income_ineq', 'median_age', 'young_percentage', 'education', 'poverty_percentage', 'unemployment', 'race_index']].apply(pd.to_numeric, errors='coerce')

    listings_df[['slope', 'median_income', 'median_property_value', 'income_ineq', 'median_age', 'young_percentage', 'education', 'poverty_percentage', 'unemployment', 'race_index']] = listings_df[['slope', 'median_property_value', 'median_income', 'income_ineq', 'median_age', 'young_percentage', 'education', 'poverty_percentage', 'unemployment', 'race_index']].apply(pd.to_numeric, errors='coerce')

    # Get rid of rows with nan values and slopes that equal to 0.0, meanining that there is an insignificant amount of reviews to reach a conclusion about that tract
    clean_reviews_df = reviews_df.dropna()
    clean_reviews_df = clean_reviews_df[clean_reviews_df['slope'] != 0.0]

    clean_listings_df = listings_df.dropna()
    clean_listings_df = clean_listings_df[clean_listings_df['slope'] != 0.0]

    # Keep track of the dropped rows in another dataframe
    reviews_nan_df = reviews_df[~reviews_df.index.isin(clean_reviews_df.index)]
    listings_nan_df = listings_df[~listings_df.index.isin(clean_listings_df.index)]

    print(clean_reviews_df.head())
    print(clean_listings_df.head())
    print(reviews_nan_df.head())
    print(listings_nan_df.head())

    return clean_reviews_df, clean_listings_df, reviews_nan_df, listings_nan_df

if __name__ == "__main__":
    reviews_df = separate_dates(argv[1])
    reviews_df = reviews_df.sort_values(by='tract_code')

    reviews_df = count_cumulative_reviews_per_month('2021-03-01', '2022-12-31', reviews_df)
    listings_df = count_new_listings_per_month('2021-03-01', '2022-12-31', argv[2])
    #hosts_df = pd.read_csv(argv[3], dtype={'tract_code': str})  
    #reviews_df['slope'] = reviews_df['slope'].map('{:,.5f}'.format)
   
    # Read in the census data and merge it into the reviews and listings dataframes
    reviews_df, listings_df = read_in_and_merge_census_data(argv[4], argv[5], reviews_df, listings_df)

    # Cleanse the data by converting the columns to numeric values and removing any rows with NaN values or values of 0
    clean_reviews_df, clean_listings_df, reviews_nan_df, listings_nan_df = cleanse_data(reviews_df, listings_df)

    # Standardize and normalize the data after dropping the rows with nan values and cleaning it up
    clean_reviews_df = plot_standardized_normal_distribution(clean_reviews_df)
    clean_listings_df = plot_standardized_normal_distribution(clean_listings_df)

    # Write the resulting dataframe containing the metrics calculated and the census data for regression to a csv file
    clean_reviews_df.to_csv(f'regression_data/{argv[4]}_census_data/{argv[4]}_reviews_data.csv', index=False, sep=',', encoding='utf-8', na_rep='NA')
    clean_listings_df.to_csv(f'regression_data/{argv[4]}_census_data/{argv[4]}_listings_data.csv', index=False, sep=',', encoding='utf-8', na_rep='NA')

    # Write the dataframe containing the rows that were dropped to a csv file
    reviews_nan_df.to_csv(f'regression_data/{argv[4]}_census_data/{argv[4]}_reviews_dropped.csv', index=False, sep=',', encoding='utf-8', na_rep='NA')
    listings_nan_df.to_csv(f'regression_data/{argv[4]}_census_data/{argv[4]}_listings_dropped.csv', index=False, sep=',', encoding='utf-8', na_rep='NA')

