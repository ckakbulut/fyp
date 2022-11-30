import csv
import os
import requests

# store the current path of this file, to be able to read relative location of scanned csv files later
script_dir = os.path.dirname(os.getcwd())


cities_dict = {}



# 6 and 7 are the latitude and longitude

def get_cities_dict():
    return cities_dict

def store_tract_id(listing_file):
    tract_dict = {}
    with open(os.path.join(script_dir, listing_file), "r") as listings:
        listing_reader = csv.reader(listings)
        listing_reader.__next__()
        for row in listing_reader:
            try:
                listing_id, lat, lng = row[0], row[6], row[7]
                # TO DO OVER HERE
            except:
                print(f"Could not parse the row with id {row[0]}")

    return tract_dict
        
        




def read_reviews(city_file):
    '''
    Reads a csv file containing reviews of a city, stores the city in a dictionary, stores the years of the reviews in a dictionary under the value of the city name, and stores the months of the reviews under the value of the years.

    i.e.  {city_name : [{2017 : [01, 02, 03 ...]}, 
                        {2018 : [01, 02, 03 ...]}]
                        }

    Parameters: 
        city_file (str): string format of csv file name being processed, in this case the reviews.csv file of each city renamed to {city_name}_reviews.csv

    Return:
        years_dict (dictionary): dictionary containing the years of the city being processed as keys and months with the corresponding listing id as a list of tuple values, ex: { 2018 : [(01, 5487182381), (05, 523315325), ...]}
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
                    years_dict[y_m_d[0]] = []  # create a new list if one doesn't exist to store the listing id together with month 
                    years_dict[y_m_d[0]].append((y_m_d[1], row[0])) #store the month as a tuple with the listing id
                else:
                    years_dict[y_m_d[0]].append((y_m_d[1], row[0])) #store the month as a tuple with the listing id 
            except ValueError:
                print(f'Could not parse the date for the row with id {row[0]}')
        # now to filter the years as we need only the data from 2018 - 2022
        for year in list(years_dict):
            if int(year) < 2018: # cast the type to int as the year keys are stored as strings
                years_dict.pop(year)
    return years_dict


seattle_years = read_reviews("fyp/csv_files/seattle_reviews.csv")
print(nyc_tracts := store_tract_id("fyp/csv_files/nyc_listings.csv"))

