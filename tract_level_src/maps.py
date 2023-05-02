''' file used to visualize geopandas maps for different census tracts and counties'''

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import census_data as cd
import statsmodels.api as sm
from sys import argv 

# takes as input the shapefile and regression data (obtained from metrics.py) and returns the merged dataframe
# plots the population for the city if plot_population set to true
def read_and_filter(shapefile, regdata, plot_population=False):
    df = gpd.read_file(f'{shapefile}')

    census_df = pd.read_csv(f'{regdata}')

    # rename the population column to have a more descriptive name
    df.rename(columns={'B01003001': 'population'}, inplace=True)

    # drop the unnecessary population margin of error column in the initial population dataframe
    df.drop(['name', 'B01003001e'], axis=1, inplace=True)

    # add the tract code column to the shapefile dataframe in order to be able to merge with census data
    # also convert the type of the tract codes to string in order to be able to merge the dataframes
    df['tract_code'] = df['geoid'].str.slice(7,18).astype(str)

    # convert the type of the tract codes to string in order to be able to merge the dataframes
    census_df['tract_code'] = census_df['tract_code'].astype(str)

    # merge the shapefile with the census information
    merged_df = df.merge(census_df, how='inner', left_on='tract_code', right_on='tract_code')

    # plot the population for the city if plot_population set to true
    if plot_population:
        merged_df.plot(legend=True, scheme='NaturalBreaks', column='population', figsize=(6,6), cmap='coolwarm')
        plt.title('Population by Census Tract')
        plt.show()

    return merged_df

def plot_correlation_matrix(merged_df, plot_scatter=False):
    correlation_df = merged_df[['median_property_value', 'median_income', 'income_ineq', 'median_age', 'young_percentage', 'education', 'poverty_percentage', 'unemployment', 'race_index']]

    # Calculate the Spearman rank correlation coefficient between pairs of columns
    corr_matrix = correlation_df.corr(method='spearman')

    # function to plot the heatmap circles
    def circle_heatmap(x, y, size, color):
        radius = size * 0.4
        circle = plt.Circle((x, y), radius, alpha=size, color=color, linewidth=0.5)
        plt.gca().add_patch(circle)

    # create a figure and axes
    norm = plt.Normalize(vmin=-1, vmax=1)
    plt.figure(figsize=(8,6))
    cmap = plt.cm.get_cmap('seismic_r')
    ax = plt.gca()
    ax.set_aspect('equal')

    # set the x and y limits of the plot and the tick and label positions
    ax.set_xlim(0, len(corr_matrix.columns))
    ax.set_ylim(0, len(corr_matrix.columns))
    ax.set_xticklabels('')
    ax.set_xticks([i+0.5 for i in range(len(corr_matrix.columns))], minor=True)
    ax.set_xticklabels([i for i in corr_matrix.columns], rotation=90, minor=True)
    ax.set_yticklabels('')
    ax.set_yticks([i+0.5 for i in range(len(corr_matrix.columns))], minor=True)
    ax.set_yticklabels([i for i in corr_matrix.columns], minor=True)

    # plot the heatmap circles
    plt.title('Pairwise Spearman Rank correlation')
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            color = cmap(corr)
            circle_heatmap(j+0.5, i+0.5, abs(corr), color)

    # create a vertical legend on the right side of the plot for the correlation matrix
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, orientation='vertical', pad=0.02, aspect=40)

    # Add the square cell outlines
    for i in range(corr_matrix.shape[0]+1):
        ax.axhline(i, color='k', linewidth=0.5)
        ax.axvline(i, color='k', linewidth=0.5)

    # show the plot
    plt.tight_layout()
    plt.show()

def plot_linear_regressions(df):
    '''
    takes in as input the merged df obtained from read_and_filter and calculates a linear regression for each tract
    '''

    # create a new dataframe to store the beta coefficients, p values and adjusted r squared values for each tract
    # the index of the dataframe will be the tract code
    #regression_results = pd.DataFrame(index=df['tract_code'], columns=['beta_coefficients', 'p_values', 'adjusted_r_squared'])

    tract_code_list = list(df['tract_code'])
    
    # Extract the dependent variable
    y = df['slope']
    
    # Extract the independent variables
    X = df[['median_income', 'median_property_value', 'income_ineq', 'median_age', 'young_percentage', 'education', 'poverty_percentage', 'unemployment', 'race_index']]

    # add a constant factor to the independent variables to account for the intercept
    X = sm.add_constant(X)

    # fit the OLS regression model
    model = sm.OLS(y, X).fit()

    print(model.summary())

    # store the summary object in a variable
    beta_coefficients = model.params
    p_values = model.pvalues
    adjusted_r_squared = model.rsquared_adj

    #return regression_results






if __name__ == "__main__":
    data = read_and_filter(argv[1], argv[2], True)
    plot_correlation_matrix(data, False)
    #regression = plot_linear_regressions(data)

    #print(regression.head(50))




