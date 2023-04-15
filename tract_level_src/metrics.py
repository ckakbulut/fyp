import pandas as pd 
import matplotlib.pyplot as plt
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
the plots/maps for the required metrics.

The file must be run from the command line with the following arguments:
    1. The merged and filtered reviews.csv file (containing the tract code associated with each review)
    2. The name of the city which the reviews are in
    3. The 5 digit FIPS code of the state and county which the city is in (used for extracting census data)
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

    df = df.sort_values(by=['tract_code'])

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
        counts_df.index = [i for i in range(1,len(counts_df.index)+1)]
        print(counts_df.index)

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

    scaler = MinMaxScaler()

    transformations = [
    ("original", lambda x: x),
    ("square root", np.sqrt),
    ("square", np.square),
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
        # if p_value >= 0.05:
        #     print(f"{col}: original")
        #     transformed_df = column.transform([np.log1p, np.square, np.sqrt])
        #     transformed_df['untransformed'] = column
        #     # move untransformed column to the front of the dataframe
        #     first_column = transformed_df.pop('untransformed')
        #     transformed_df.insert(0, 'untransformed', first_column)
        #     transformed_df.hist(bins = 20, figsize=(10,10), layout=(3,2), edgecolor='black')
        #     plt.suptitle(f'Histograms of Transformed Data for {col}', size=16)
        #     plt.show()
        #     df[col] = (df[col] - df[col].mean()) / df[col].std()
        #     df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))
        #     continue
        
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
        print(f"{col}: {best_transform[0]}", "p-value: ", best_p_value)
        
        transformed_df = column.transform([np.log1p, np.square, np.sqrt])
        transformed_df['untransformed'] = column

        # move untransformed column to the front of the dataframe
        first_column = transformed_df.pop('untransformed')
        transformed_df.insert(0, 'untransformed', first_column)
        transformed_df.hist(bins = 20, figsize=(10,10), layout=(3,2), edgecolor='black')
        plt.suptitle(f'Histograms of Transformed Data for {col}', size=16)
        plt.show()

        # Standardize the data
        df[col] = (df[col] - np.mean(df[col])) / np.std(df[col])
        #df[col] = (df[col] - df[col].mean()) / df[col].std()

        # normalize the data between 0 and 1 (TODO - MIGHT NEED TO CHANGE THIS)
        df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))
    
    return df


def plot_linear_regressions(df):
    '''
    takes in as input the slopes_df calculated frfom the count_cumulative_reviews_per_month function and plots a linear regression for each tract
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
    '''function used to create the final dataframe containing the census data for each tract the slope (the metric we defined) for each tract on each row'''


    # Convert the index of df_a to type str
    df_a['tract_code'] = df_a['tract_code'].astype(int)
    df_a['tract_code'] = df_a['tract_code'].astype(str)


    df_b.index.name = 'tract_code'
    df_b.index = df_b.index.astype(str)

    # Merge the two dataframes based on the index of df_a and column names of df_b_t
    # we merge on the inner to only keep the tracts that are in both dataframes
    new_df = df_a.merge(df_b, how='inner', left_on='tract_code', right_on='tract_code')

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
    slopes_df = count_cumulative_reviews_per_month('2021-03-01', '2022-10-01', reviews_df)
    slopes_df['slope'] = slopes_df['slope'].map('{:,.5f}'.format)
   
    # median_property_data = census_data.single_row_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_median_property.csv', f'{argv[3]}')
    # median_income_data = census_data.single_row_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_median_income.csv', f'{argv[3]}')
    # income_ineq_data = census_data.single_row_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_income_ineq.csv', f'{argv[3]}')
    # median_age_data = census_data.median_age_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_median_age.csv', f'{argv[3]}')
    # age_data = census_data.age_data(f'~/Desktop/{argv[2]}_census_data/orleans_age.csv', f'22071')
    # education_data = census_data.educational_attainment_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_education.csv', f'{argv[3]}')
    # poverty_data = census_data.percentage_poverty_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_poverty.csv', f'{argv[3]}')
    # unemployment_data = census_data.unemployment_rate_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_unemployment.csv', f'{argv[3]}')
    # race_data = census_data.race_diversity_data(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_race.csv', f'{argv[3]}')

    # # Merge the slopes_df with the median property value data
    # slopes_df = merge_dataframes(slopes_df, median_property_data, 'median_property_value')
    # slopes_df = merge_dataframes(slopes_df, median_income_data, 'median_income')
    # slopes_df = merge_dataframes(slopes_df, income_ineq_data, 'income_ineq')
    # slopes_df = merge_dataframes(slopes_df, median_age_data, 'median_age')
    # slopes_df = merge_dataframes(slopes_df, age_data, 'young_percentage')
    # slopes_df = merge_dataframes(slopes_df, education_data, 'education')
    # slopes_df = merge_dataframes(slopes_df, poverty_data, 'poverty_percentage')
    # slopes_df = merge_dataframes(slopes_df, unemployment_data, 'unemployment')
    # slopes_df = merge_dataframes(slopes_df, race_data, 'race_index')

    # slopes_df = slopes_df.apply(pd.to_numeric, errors='coerce')

    # slopes_df = plot_standardized_normal_distribution(slopes_df)
    print(slopes_df.head())

    #slopes_df.to_csv(f'~/Desktop/{argv[2]}_census_data/{argv[2]}_regdata.csv', index=False, sep=',', encoding='utf-8')
    #plot_linear_regressions(slopes_df)
