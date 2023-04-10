import pandas as pd 
import matplotlib.pyplot as plt
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import normaltest, shapiro
from sklearn.linear_model import LinearRegression
from datetime import datetime
import re
import census_data

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


def count_cumulative_reviews_per_month(start_date, end_date, df):
    # convert start_date and end_date to datetime objects
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)

    # create a new dataframe with date as the index and cumulative count as the column
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    counts_df = pd.DataFrame(index=date_range, columns=['cumulative_count'])

    # iterate through each month and count the cumulative number of reviews (FOR THE ENTIRE CITY / ALL TRACTS)
    for month_start in date_range:
        month_end = month_start + pd.offsets.MonthEnd(0)
        cum_count = len(df[(df['date'] >= start_date) & (df['date'] <= month_end)])
        counts_df.loc[month_start, 'cumulative_count'] = cum_count

    # create a new dataframe to store the slope for each tract
    slopes_df = pd.DataFrame(columns=['tract_code', 'slope'])

    # iterate through each tract and plot the line of best fit
    for tract in df['tract_code'].unique():
        tract_df = df[df['tract_code'] == tract]

        # create a new dataframe with date as the index and cumulative count as the column
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        counts_df = pd.DataFrame(index=date_range, columns=['cumulative_count'])

        # iterate through each month and count the cumulative number of reviews
        for month_start in date_range:
            month_end = month_start + pd.offsets.MonthEnd(0)
            cum_count = len(tract_df[(tract_df['date'] >= start_date) & (tract_df['date'] <= month_end)])
            counts_df.loc[month_start, 'cumulative_count'] = cum_count

        # store the dates in ISO format before converting them to ordinal which will be used to label the x-axis on the plot
        plot_X = counts_df.index

        # convert the dates to ordinal in order to be able to use them in the linear regression
        counts_df.index = pd.to_datetime(counts_df.index)
        counts_df.index = counts_df.index.map(datetime.toordinal)

        # fit a linear regression to the counts_df
        X = counts_df.index.values.reshape(-1, 1)
        y = counts_df['cumulative_count'].values.reshape(-1,1)
        # model = sm.OLS(y, X)
        # results = model.fit()
        lr = LinearRegression().fit(X, y)

        # plot the line of best fit
        plt.plot(plot_X, lr.predict(X), label=f'Tract {tract}')
        # print("The y values are: ", lr.predict(X))
        # print("Score is ", lr.score(X, y))
        # print(f"Slope received from scikit is {lr.coef_[0][0]}")

        # store the slope in the slopes_df
        slopes_df = slopes_df.append({'tract_code': tract, 'slope': lr.coef_[0][0]}, ignore_index=True)

    # set the x-axis label and legend
    plt.xlabel('Date')
    plt.legend()

    # show the plot
    plt.show()

    # return the slopes_df
    return slopes_df


def count_new_listings_per_month(start_date, end_date, df):
    #convert the start_date and end_date to datetime objects
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)

    # create a new dataframe with date as the index and count as the column
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    counts_df = pd.DataFrame(index=date_range, columns=['count'])

    # count the number of new hosts added each month between the given time interval (FOR ENTIRE CITY / ALL TRACTS)
    city_df = df.groupby(pd.Grouper(key='date', freq='M'))['listing_id'].nunique().reset_index()
    city_df = city_df.rename(columns={'listing_id': 'new_hosts'})
    city_df = city_df[(city_df['date'] >= start_date) & (city_df['date'] <= end_date)]

    # create a new dataframe to store the slope for each tract
    slopes_df = pd.DataFrame(columns=['tract_code', 'slope'])


    # iterate through each tract and plot the line of best fit
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

        # convert the dates to ordinal in order to be able to use them in the linear regression
        counts_df.index = pd.to_datetime(counts_df.index)
        counts_df.index = counts_df.index.map(datetime.toordinal)

        # fit a linear regression to the counts_df
        X = counts_df.index.values.reshape(-1, 1)
        y = counts_df['count'].values.reshape(-1,1)
        lr = LinearRegression().fit(X, y)
        
        # plot the line of best fit
        plt.plot(plot_X, lr.predict(X), label=f'Tract {tract}')
        print("INTERCEPT IS: ", lr.intercept_)
        print("The ordinal times are: ", counts_df.index)
        print("The y values are: ", lr.predict(X))
        print("Score is ", lr.score(X, y))
        
        # store the slope in the slopes_df
        slopes_df = slopes_df.append({'tract_code': tract, 'slope': lr.coef_[0][0]}, ignore_index=True)
    
    # set the x-axis label and legend
    plt.xlabel('Date')
    plt.legend()
    
    # show the plot
    plt.show()
    
    # return the slopes_df
    return slopes_df

def plot_standardized_normal_distribution(df):
    ''' 
    takes in as input the slopes_df calculated from the count_cumulative_reviews_per_month function and returns a normally distributed version of the slopes inside the slopes_df
    '''

    transformations = [
    ("original", lambda x: x),
    ("square root", np.sqrt),
    ("square", np.square),
    ("log", np.log1p)
    ]

    columns = df.columns

    alpha = 0.05

    for col in columns:
        # extract the column of interest from the dataframe
        if col == 'tract_code':
            continue
        column = df[col]
        #Check if the data is already normally distributed
        _, p_value = shapiro(df[col])
        if p_value >= 0.05:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
            print(f"{col}: original")
            continue
        
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
        
        # df[col].hist(bins = 20, figsize=(10,10), edgecolor='black')
        # plt.suptitle(f'Histogram of {col} Data', size=16)
        # plt.show()
        transformed_df = column.transform([np.log1p, np.square, np.sqrt])
        transformed_df['untransformed'] = column
        # move untransformed column to the front
        first_column = transformed_df.pop('untransformed')
        transformed_df.insert(0, 'untransformed', first_column)
        transformed_df.hist(bins = 20, figsize=(10,10), layout=(3,2), edgecolor='black')
        plt.suptitle(f'Histograms of Transformed Data for {col}', size=16)
        plt.show()

        # Standardize the data
        df[col] = (df[col] - df[col].mean()) / df[col].std()
        # normalize the data between 0 and 1
        #df[col] = (df[col] - np.min(df[col]) / (np.max(df[col]) - np.min(df[col])))
    
    return df


def plot_linear_regressions(df):
    '''
    takes in as input the slopes_df calculated from the count_cumulative_reviews_per_month function and plots a linear regression for each tract
    '''
    # iterate through each tract and plot the line of best fit
    for tract in df['tract_code'].unique():
        tract_df = df[df['tract_code'] == tract]

        # store the indepdent variables in a separate dataframe to be able to use them later
        x_tract = tract_df.drop(columns=['tract_code', 'slope'])
        x_values = x_tract.values.reshape(-1,1)

        print(tract_df['slope'].values.reshape(-1,1))

        # fit a linear regression to the counts_df
        X = tract_df[['median_property_value', 'median_income', 'income_ineq', 'median_age', 'young_percentage', 'education', 'poverty_percentage', 'unemployment', 'race_index']]
        y = tract_df['slope']
        lr = LinearRegression().fit(X, y)


def merge_dataframes(df_a, df_b, new_column_name):
    # Convert the index of df_a to type str
    df_a['tract_code'] = df_a['tract_code'].astype(int)
    df_a['tract_code'] = df_a['tract_code'].astype(str)


    df_b.index.name = 'tract_code'
    df_b.index = df_b.index.astype(str)

    # Merge the two dataframes based on the index of df_a and column names of df_b_t
    new_df = df_a.merge(df_b, how='left', left_on='tract_code', right_on='tract_code')

    #Rename the column with the merged values to new_column_name
    new_df.rename(columns={0: f'{new_column_name}'}, inplace=True)

    new_df[new_column_name] = new_df[new_column_name].apply(lambda x: re.sub(r'[^0-9\.]', '', str(x)) if x else x)
    new_df[new_column_name] = new_df[new_column_name].apply(lambda x: float(x) if x else x)

    return new_df

if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 150)
    reviews_df = separate_dates(argv[1])
    slopes_df = count_cumulative_reviews_per_month('2021-03-01', '2022-08-01', reviews_df)
    slopes_df['slope'] = slopes_df['slope'].map('{:,.5f}'.format)
   
    median_property_data = census_data.single_row_data('~/Desktop/seattle_census_data/seattle_median_property_value.csv', '53033')
    median_income_data = census_data.single_row_data('~/Desktop/seattle_census_data/seattle_median_household_income.csv', '53033')
    income_ineq_data = census_data.single_row_data('~/Desktop/seattle_census_data/seattle_income_ineq.csv', '53033')
    
    median_age_data = census_data.median_age_data('~/Desktop/seattle_census_data/seattle_median_age.csv', '53033')
    age_data = census_data.age_data('~/Desktop/seattle_census_data/seattle_age.csv', '53033')
    education_data = census_data.educational_attainment_data('~/Desktop/seattle_census_data/seattle_educational_attainment.csv', '53033')
    poverty_data = census_data.percentage_poverty_data('~/Desktop/seattle_census_data/seattle_percent_poverty.csv', '53033')
    unemployment_data = census_data.unemployment_rate_data('~/Desktop/seattle_census_data/seattle_unemployment.csv', '53033')
    race_data = census_data.race_diversity_data('~/Desktop/seattle_census_data/seattle_race.csv', '53033')

    # Merge the slopes_df with the median property value data
    slopes_df = merge_dataframes(slopes_df, median_property_data, 'median_property_value')
    slopes_df = merge_dataframes(slopes_df, median_income_data, 'median_income')
    slopes_df = merge_dataframes(slopes_df, income_ineq_data, 'income_ineq')
    slopes_df = merge_dataframes(slopes_df, median_age_data, 'median_age')
    slopes_df = merge_dataframes(slopes_df, age_data, 'young_percentage')
    slopes_df = merge_dataframes(slopes_df, education_data, 'education')
    slopes_df = merge_dataframes(slopes_df, poverty_data, 'poverty_percentage')
    slopes_df = merge_dataframes(slopes_df, unemployment_data, 'unemployment')
    slopes_df = merge_dataframes(slopes_df, race_data, 'race_index')

    slopes_df = slopes_df.apply(pd.to_numeric, errors='coerce')

    #print(slopes_df.head(200))

    slopes_df = plot_standardized_normal_distribution(slopes_df)
    plot_linear_regressions(slopes_df)
