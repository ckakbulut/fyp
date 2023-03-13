import pandas as pd 
import matplotlib.pyplot as plt
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

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
        tract_df = reviews_count.loc[code]
        plt.xlabel('Time (Months)')
        plt.ylabel('Monthly Review Count')
        for year in tract_df.columns:
            plt.errorbar(tract_df.index, tract_df[year], yerr=tract_df[year].sem(), capsize=5, capthick=2, label=year, ecolor='gray')
        
        plt.title(f'Monthly Review Count for Tract Code {code}')
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
        plt.title(f'Yearly Review Count for Tract Code {code}')
        #plt.show()
        plt.savefig(f'{code}.png', format='png', bbox_inches='tight')
        plt.close()

def count_reviews_per_month(start_date, end_date, df):
    # convert start_date and end_date to datetime objects
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)
    
    # create a new dataframe with date as the index and count as the column
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    counts_df = pd.DataFrame(index=date_range, columns=['count'])
    
    # iterate through each month and count the number of reviews
    for month_start in date_range:
        month_end = month_start + pd.offsets.MonthEnd(0)
        month_count = len(df[(df['date'] >= month_start) & (df['date'] <= month_end)])
        counts_df.loc[month_start, 'count'] = month_count
    
    # create a new dataframe to store the slope for each tract
    slopes_df = pd.DataFrame(columns=['tract_code', 'slope'])
    
    # iterate through each tract and plot the line of best fit
    for tract in df['tract_code'].unique():
        tract_df = df[df['tract_code'] == tract]
        
        # create a new dataframe with date as the index and count as the column
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        counts_df = pd.DataFrame(index=date_range, columns=['count'])
        
        # iterate through each month and count the number of reviews
        for month_start in date_range:
            month_end = month_start + pd.offsets.MonthEnd(0)
            month_count = len(tract_df[(tract_df['date'] >= month_start) & (tract_df['date'] <= month_end)])
            counts_df.loc[month_start, 'count'] = month_count


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

def sample1(start_date, end_date, df):
    # convert start_date and end_date to datetime objects
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)
    
    # create a new dataframe with date as the index and count as the column
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    counts_df = pd.DataFrame(index=date_range, columns=['count'])
    
    # iterate through each month and count the number of reviews for ALL THE TRACTS
    for month_start in date_range:
        month_end = month_start + pd.offsets.MonthEnd(0)
        month_count = len(df[(df['date'] >= month_start) & (df['date'] <= month_end)])
        counts_df.loc[month_start, 'count'] = month_count
    
    # create a new dataframe to store the slope for each tract
    slopes_df = pd.DataFrame(columns=['tract_code', 'slope'])
    
    # iterate through each tract and plot the line of best fit
    counter = 0
    for tract in df['tract_code'].unique():
        if(counter == 1):
            break
        counter += 1
        tract_df = df[df['tract_code'] == tract]
        
        # create a new dataframe with date as the index and count as the column
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        counts_df = pd.DataFrame(index=date_range, columns=['count'])
        
        # iterate through each month and count the number of reviews
        for month_start in date_range:
            month_end = month_start + pd.offsets.MonthEnd(0)
            month_count = len(tract_df[(tract_df['date'] >= month_start) & (tract_df['date'] <= month_end)])
            counts_df.loc[month_start, 'count'] = month_count


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

def new_listings_per_month(df):
    pass

def new_listings_per_year(df):
    pass




if __name__ == "__main__":
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)
    pd.set_option('display.float_format', lambda x: '%.0f' % x if x.name == 'tract_code' else '%.6f' % x)
    reviews_df = separate_dates(argv[1])
    #total_reviews_per_month(reviews_df)
    #slopes_df = sample1('2022-03-01', '2022-08-01', reviews_df)
    slopes_df = count_reviews_per_month('2021-03-01', '2022-08-01', reviews_df)
    print(slopes_df)
    print(slopes_df['slope'].max())
    #total_reviews_per_year(reviews_df)
    #cumulative_reviews_per_month(reviews_df)
    #cumulative_reviews_per_year(reviews_df)
