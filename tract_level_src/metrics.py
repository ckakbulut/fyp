import pandas as pd 
import matplotlib.pyplot as plt
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
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
    2. The merged listings.csv file (containing the tract code associated with each listing) -> used to calculate cumulative # new hosts per month
    3. The name of the city which the reviews are in
    4. The 5 digit FIPS code of the state and county which the city is in (used for extracting census data)
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
    ax.set_xticklabels('')
    ax.set_xticks([i+0.5 for i in range(len(counts_df_city.index))], minor=True)
    ax.set_xticklabels([i for i in counts_df_city.index], rotation=45, fontsize=6, minor=True)
    plt.show()

    # create a new dataframe to store the slope for each tract
    slopes_df = pd.DataFrame(columns=['tract_code', 'review_slope'])

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


def count_new_listings_per_month(start_date, end_date, df):
    #convert the start_date and end_date to datetime objects
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)

    # create a new dataframe with date as the index and count as the column
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    counts_df = pd.DataFrame(index=date_range, columns=['count'])

    # count the number of new hosts added each month between the given time interval (FOR ENTIRE CITY / ALL TRACTS)
    city_df = df.groupby(pd.Grouper(key='date', freq='M'))['listing_id'].nunique().reset_index()
    city_df = city_df.rename(columns={'listing_id': 'new_hosts'})
    city_df = city_df[(city_df['date'] >= start_date) & (city_df['date'] <= end_date)]

    # create a new dataframe to store the slope for each tract
    slopes_df = pd.DataFrame(columns=['tract_code', 'hosts_slope'])


    # iterate through each tract and count the number of new hosts added each month between the given time interval
    for tract in df['tract_code'].unique():
        tract_df = df[df['tract_code'] == tract]
        
        # create a new dataframe with date as the index and count as the column
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        counts_df = pd.DataFrame(index=date_range, columns=['count'])
        
        # iterate through each month and count the number of reviews
        for month_start in date_range:
            month_end = month_start + pd.offsets.MonthEnd(0)
            month_count = len(tract_df[(tract_df['date'] >= month_start) & (tract_df['date'] <= month_end)].groupby(pd.Grouper(key='date', freq='M'))['listing_id'].nunique().reset_index())
            counts_df.loc[month_start, 'count'] = month_count
        
        # fill in any missing values with 0
        counts_df = counts_df.fillna(0)

        # store the dates in ISO format before converting them to ordinal which will be used to label the x-axis on the plot
        plot_X = counts_df.index

         # convert the dates to a time series in order to be able to use them in the linear regression
        counts_df.index = pd.to_datetime(counts_df.index)
        counts_df.index = [i for i in range(1,len(counts_df.index)+1)]

        # fit a linear regression to the counts_df
        X = counts_df.index.values.reshape(-1, 1)
        y = counts_df['count'].values.reshape(-1,1)
        lr = LinearRegression().fit(X, y)
        
        # store the slope in the slopes_df
        slopes_df = slopes_df.append({'tract_code': tract, 'slope': lr.coef_[0][0]}, ignore_index=True)
    
    # return the slopes_df
    return slopes_df

def plot_standardized_normal_distribution(df):
    ''' 
    takes in as input the slopes_df calculated from the count_cumulative_reviews_per_month function and returns a normally distributed version of the slopes inside the slopes_df
    '''

    scaler = MinMaxScaler()

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

        # move untransformed column to the front of the dataframe
        first_column = transformed_df.pop('untransformed')
        transformed_df.insert(0, 'untransformed', first_column)
        transformed_df.hist(bins = 20, figsize=(10,10), layout=(3,2), edgecolor='black')
        plt.suptitle(f'Histograms of Transformed Data for {col}', size=16)
        plt.show()

        if col != 'slope':
            # Standardize the data
            df[col] = (df[col] - np.mean(df[col])) / np.std(df[col])

        # normalize the data between 0 and 1
        #df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))
    
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

if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    reviews_df = separate_dates(argv[1])
    reviews_df = reviews_df.sort_values(by='tract_code')
    count_absolute_reviews_per_month('2018-01-01', '2022-12-01', reviews_df)
    #slopes_df = count_cumulative_reviews_per_month('2018-01-01', '2022-12-01', reviews_df)
    # slopes_df['slope'] = slopes_df['slope'].map('{:,.5f}'.format)
   
    # # Read in the census data
    # median_property_data = census_data.single_row_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_median_property.csv', f'{argv[3]}')
    # median_income_data = census_data.single_row_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_median_income.csv', f'{argv[3]}')
    # income_ineq_data = census_data.single_row_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_income_ineq.csv', f'{argv[3]}')
    # median_age_data = census_data.median_age_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_median_age.csv', f'{argv[3]}')
    # age_data = census_data.age_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_age.csv', f'{argv[3]}')
    # education_data = census_data.educational_attainment_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_education.csv', f'{argv[3]}')
    # poverty_data = census_data.percentage_poverty_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_poverty.csv', f'{argv[3]}')
    # unemployment_data = census_data.unemployment_rate_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_unemployment.csv', f'{argv[3]}')
    # race_data = census_data.race_diversity_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_race.csv', f'{argv[3]}')

    # # Merge the slopes_df with the census data dataframes
    # slopes_df = merge_dataframes(slopes_df, median_property_data, 'median_property_value')
    # slopes_df = merge_dataframes(slopes_df, median_income_data, 'median_income')
    # slopes_df = merge_dataframes(slopes_df, income_ineq_data, 'income_ineq')
    # slopes_df = merge_dataframes(slopes_df, median_age_data, 'median_age')
    # slopes_df = merge_dataframes(slopes_df, age_data, 'young_percentage')
    # slopes_df = merge_dataframes(slopes_df, education_data, 'education')
    # slopes_df = merge_dataframes(slopes_df, poverty_data, 'poverty_percentage')
    # slopes_df = merge_dataframes(slopes_df, unemployment_data, 'unemployment')
    # slopes_df = merge_dataframes(slopes_df, race_data, 'race_index')

    # slopes_df[['slope', 'median_income', 'median_property_value', 'income_ineq', 'median_age', 'young_percentage', 'education', 'poverty_percentage', 'unemployment', 'race_index']] = slopes_df[['slope', 'median_property_value', 'median_income', 'income_ineq', 'median_age', 'young_percentage', 'education', 'poverty_percentage', 'unemployment', 'race_index']].apply(pd.to_numeric, errors='coerce')

    # # Get rid of rows with nan values and slopes that equal to 0.0, meanining that there is an insignificant amount of reviews to reach a conclusion about that tract
    # clean_df = slopes_df.dropna()
    # clean_df = clean_df[clean_df['slope'] != 0.0]

    # # Keep track of the dropped rows in another dataframe
    # nan_df = slopes_df[~slopes_df.index.isin(clean_df.index)]

    # # Standardize and normalize the data after dropping the rows with nan values and cleaning it up
    # clean_df = plot_standardized_normal_distribution(clean_df)

    # print(clean_df.head(20))

    # # Write the resulting dataframe containing the metrics calculated and the census data for regression to a csv file
    # clean_df.to_csv(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_regdata.csv', index=False, sep=',', encoding='utf-8', na_rep='NA')

    # # Write the dataframe containing the rows that were dropped to a csv file
    # nan_df.to_csv(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_dropped.csv', index=False, sep=',', encoding='utf-8', na_rep='NA')
