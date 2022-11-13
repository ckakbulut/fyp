import csv
import os

# store the current path of this file, to be able to read relative location of scanned csv files later
script_dir = os.path.dirname(__file__)


cities_dict = {}




def read_reviews(city_file):
    '''
    Reads a csv file containing reviews of a city, stores the city in a dictionary, stores the years of the reviews in a dictionary under the value of the city name, and stores the months of the reviews under the value of the years.

    i.e.  {city_name : [{2017 : [01, 02, 03 ...]}, 
                        {2018 : [01, 02, 03 ...]}]
                        }

    Parameters: 
        city_file (str): string format of csv file name being processed, in this case the reviews.csv file of each city renamed to {city_name}_reviews.csv

    Returns:
        years_dict (dictionary): dictionary containing the years of the city being processed as keys and months as list of values 
    '''
    with open(os.path.join(script_dir, city_file), "r") as file: # since all the csv files will be in the same folder as this module, get relative path of csv file
        city_name = city_file.split("-")[0] # retrieve the name of the city being processed from the csv file name, such as seattle from seattle_reviews.csv
        csvreader = csv.reader(file)
        csvreader.__next__() # skip the first row of the csv file with the names of the columns
        cities_dict[city_name] = {} # create a new dictionary for each city with corresponding year
        years_dict = cities_dict.get(city_name) 
        for row in csvreader:
            try:
                y_m_d = row[2].split("-") # split the dates into year, month and day according to isoformat
                # store the month corresponding to each year of the review
                if years_dict.get(y_m_d[0]) == None:
                    years_dict[y_m_d[0]] = []  
                    years_dict[y_m_d[0]].append(y_m_d[1])
                if years_dict.get(y_m_d[0]) != None:
                    years_dict[y_m_d[0]].append(y_m_d[1])
            except ValueError:
                print(f'Could not parse the date for the row with id {row[0]}')
        # now to filter the years as we need only the data from 2018 - 2022
        for year in list(years_dict):
            if int(year) < 2018: # cast the type to int as the year keys are stored as strings
                years_dict.pop(year)
    return years_dict


seattle_years = read_reviews("./csv_files/seattle_reviews.csv")

print(seattle_years.keys())

