import pandas as pd
import re

# auxiliary function to add zeros in filling the FIPS code for census data columns containing tract codes
def fill_zeros(string, length):
    if len(string) < length:
        return fill_zeros("0" + string, length)
    else:
        return string

# used to reshape census data column headers that include one column per tract and contain the tract code in the column name
def rename_census_data_columns(df, county_code):
    # pd.set_option('display.max_columns', None)

    # store the columns 
    columns = df.columns

    # for each column name, remove any non digit characters
    df.columns = [re.sub("[^0-9\.]", "", col) for col in columns]

    # if the column name doesn't contain a dot, add two zeros at the end as described in this article
    # https://www.easidemographics.com/trshelp/html/census_tract_2.htm
    df.columns = [col + '00' if "." not in col else col for col in df.columns]

    # remove the decimal point from the column names
    df.columns = [re.sub("\.", "", col) for col in df.columns]

    # if the length of the column name is less than 6 (FIPS tract code max length), add zeros to the front
    df.columns = [fill_zeros(col, 6) for col in df.columns]

    # add the county code to the front of the column name to get the full 11 digits FIPS code
    df.columns = [county_code + col for col in df.columns]

    return df

def single_row_data(filename, county_code):
    # used in extracting data from census files where the required data is present in a single row in the downloaded csv file (such as median property value)

    df = pd.read_csv(filename)

    # store the columns 
    columns = df.columns

    # for each column name, remove any non digit and non decimal point characters to only be left with the tract code
    df.columns = [re.sub("[^0-9\.]", "", col) for col in columns]

    # if the column name doesn't contain a dot, add two zeros at the end as described in this article
    # https://www.easidemographics.com/trshelp/html/census_tract_2.htm
    df.columns = [col + '00' if "." not in col else col for col in df.columns]

    # remove the decimal point from the column names
    df.columns = [re.sub("\.", "", col) for col in df.columns]

    # if the length of the column name is less than 6, add zeros to the front
    df.columns = [fill_zeros(col, 6) for col in df.columns]

    # add the county code to the front of the column name to get the full 11 digits FIPS code
    df.columns = [county_code + col for col in df.columns]
    
    # transpose the dataframe so that the tract codes are the index
    df = df.transpose()

    # drop all rows that contain the value 0 as we can infer they contain missing information
    df = df[df[0] != 0.0]

    return df


def median_age_data(filename, county_code):

    df = pd.read_csv(filename)

    # return the row containing the median age data for the total population (all genders)
    df = df.iloc[[1]]

    rename_census_data_columns(df, county_code)

    df = df.transpose()  

    # rename the column to 0 so that it can be renamed to median age in metrics.py 
    df.rename(columns={1: 0}, inplace=True)

    # drop all rows that contain the value 0 as we can infer they contain missing information
    df = df[df[0] != 0.0]

    return df


def age_data(filename, county_code):

    df = pd.read_csv(filename)
    
    regex = re.compile(r'(Label|Percent\!\!Estimate)', re.IGNORECASE)

    df = df.filter(regex=regex, axis=1)

    # make the first column the index
    df = df.set_index(df.columns[0])

    # strip the index (the labels) of any whitespace (caused due to indentation in the csv file)
    df.index = df.index.str.strip()

    # rename columns according to county code (to get the full 11 digit FIPS code)
    rename_census_data_columns(df, county_code)

    # get the sum of the 20 to 34 year olds (young people)
    df = df.loc[['20 to 24 years', '25 to 29 years', '30 to 34 years']]

    # divide by 100 to get the percentage as a decimal value
    df = df.apply(lambda col: col.str.extract(r'(\d+\.\d+)%').astype(float).sum()) / 100

    df = df.transpose()

    # drop all rows that contain the value 0 as we can infer they contain missing information
    df = df[df[0] != 0.0]

    return df


def educational_attainment_data(filename, county_code):
    
    df = pd.read_csv(filename)
    
    regex = re.compile(r'(Label|Total\!\!Estimate)', re.IGNORECASE)

    df = df.filter(regex=regex, axis=1)

    # make the first column the index
    df = df.set_index(df.columns[0])

    # strip the index (the labels) of any whitespace (caused due to indentation in the csv file)
    df.index = df.index.str.strip()

    # rename columns according to county code (to get the full 11 digit FIPS code)
    rename_census_data_columns(df, county_code)

    divide_total = df.loc[['Population 18 to 24 years', 'Population 25 years and over']]

    df = df.loc[['18-24 Bachelor\'s degree or higher', '25 Above Bachelor\'s degree or higher']]

    divide_total = divide_total.apply(lambda col: col.str.split(',').str.join('').astype(int).sum()).reset_index()

    df = df.apply(lambda col: col.str.split(',').str.join('').astype(int).sum()).reset_index()

    # get the percentage of people with a bachelor's degree or higher educational status
    df[0] = df[0] / divide_total[0]

    # set the index to the index column instead of tract_code
    df.set_index("index", inplace=True)

    # drop all rows that contain the value 0 as we can infer they contain missing information
    df = df[df[0] != 0.0]

    return df


def percentage_poverty_data(filename, county_code):

    df = pd.read_csv(filename)
    
    regex = re.compile(r'(Label|Percent)', re.IGNORECASE)

    df = df.filter(regex=regex, axis=1)

    # make the first column the index
    df = df.set_index(df.columns[0])

    rename_census_data_columns(df, county_code)

    df = df.loc[['Population for whom poverty status is determined']]

    df = df.apply(lambda col: col.str.extract(r'(\d+\.\d+)%').astype(float).sum()) / 100

    # transpose the dataframe so that the tract codes are the index
    df = df.transpose()

    # drop all rows that contain the value 0 as we can infer they contain missing information
    df = df[df[0] != 0.0]

    return df

def unemployment_rate_data(filename, county_code):

    df = pd.read_csv(filename)
    
    regex = re.compile(r'(Label|Unemployment)', re.IGNORECASE)

    df = df.filter(regex=regex, axis=1)

    # make the first column the index
    df = df.set_index(df.columns[0])

    rename_census_data_columns(df, county_code)

    df = df.loc[['Population 16 years and over']]

    df = df.apply(lambda col: col.str.extract(r'(\d+\.\d+)%').astype(float).sum()) / 100

    df = df.transpose()

    # drop all rows that contain the value 0 as we can infer they contain missing information
    df = df[df[0] != 0.0]
    
    return df



def race_diversity_data(filename, county_code):

    df = pd.read_csv(filename)

    df = df.set_index(df.columns[0])


    rename_census_data_columns(df, county_code)

    races = ['Hispanic or Latino', 'White', 'Black or African American', 'Asian', 'Native Hawaiian and Other Pacific Islander', 'American Indian and Alaska Native', 'Some Other Race']

    df.index = df.index.str.strip()

    # delete the row with label "Not Hispanic or Latino" to make dataframe processing for Simpson diversity index easier
    df = df.drop("Not Hispanic or Latino:", axis=0)

    # delete any rows that do not contain relevant information (such as indicating total population for x amount of races)
    df = df.drop(df.index[df.index.str.contains("Population")], axis=0)

    # create new df which will later on store our simpson index values
    index_df = pd.DataFrame(index=df.columns, columns=['race_index'])

    tracts_simpson = []

    for tract_code in df.columns:
        race_population = {}
        for index in df.index:
            for race in races:
                if race in index:
                    count = df.loc[index, tract_code]
                    if race not in race_population:
                        if (type(count) == str):
                            race_population[race] = int(count.replace(',', ''))
                        else:
                            race_population[race] = count
                    else:
                        if (type(count) == str):
                            race_population[race] += int(count.replace(',', ''))
                        else:
                            race_population[race] += count
        N = sum(race_population.values())
        if N == 0:
            simpson_index = 0
        else:
            simpson_index = 1 - sum(n * (n-1) for n in race_population.values()) / (N * (N-1))
        index_df.loc[tract_code, 'race_index'] = simpson_index
        tracts_simpson.append((tract_code, race_population))

    # drop all rows that contain the value 0 as we can infer they contain missing information
    index_df = index_df[index_df['race_index'] != 0.0]

    return index_df




if __name__ == "__main__":
    median_age_data('~/Desktop/oakland_census_data/oakland_median_age.csv', '06001')
    age_data('~/Desktop/oakland_census_data/oakland_age.csv', '06001')
    educational_attainment_data('~/Desktop/oakland_census_data/oakland_education.csv', '06001')
    percentage_poverty_data('~/Desktop/oakland_census_data/oakland_poverty.csv', '06001')
    unemployment_rate_data('~/Desktop/oakland_census_data/oakland_unemployment.csv', '06001')
    race_diversity_data('~/Desktop/oakland_census_data/oakland_race.csv', '06001')
    median_property_data = single_row_data(f'~/Desktop/oakland_census_data/oakland_median_property.csv', '06001')
    median_income_data = single_row_data(f'~/Desktop/oakland_census_data/oakland_median_income.csv', '06001')
    income_ineq_data = single_row_data(f'~/Desktop/oakland_census_data/oakland_income_ineq.csv', '06001')
