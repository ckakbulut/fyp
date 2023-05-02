from sys import argv
import requests
import csv

wanted_columns= ["id", "tract_code", "host_id", "host_since", "host_total_listings_count", "reviews_per_month", "first_review", "last_review", "number_of_reviews", "calendar_last_scraped"] 

def get_tract_code(latitude,longitude):
    api = 'https://geo.fcc.gov/api/census/area'
    try: 
        response = requests.get(f'{api}?lat={latitude}&lon={longitude}&censusYear=2020&format=json').json()
        block_fips = response["results"][0]["block_fips"]
        census_tract = block_fips[:11]
    except Exception as e: 
        print(e)
    return census_tract

def tracts_to_csv(listings, skiprows=0, mode='w'):
    with open(listings, 'r') as file:
        with open('census_tract_listings.csv', mode) as output:
            csvreader = csv.DictReader(file)
            writer = csv.DictWriter(output, wanted_columns)
            if(int(skiprows) == 0):
                writer.writeheader()
            count = 0
            for row in csvreader: 
                # used to skip the first n rows of the csv file if the script is interrupted
                if count < int(skiprows):
                    count += 1
                    continue
                tract_code = get_tract_code(row['latitude'], row['longitude'])
                print(tract_code)
                writer.writerow({"id": row['id'], "tract_code": tract_code, "host_id": row['host_id'], "host_since": row['host_since'], "host_total_listings_count": row['host_total_listings_count'], "reviews_per_month": row['reviews_per_month'], "first_review": row['first_review'], "last_review": row['last_review'] , "number_of_reviews": row['number_of_reviews'], "calendar_last_scraped": row['calendar_last_scraped']}) 



''' takes in as arguments 
1) the original listings.csv file, 
2) the number of rows to skip (if the script is interrupted)
3) the mode (w for write, a for append) 
'''

if __name__ == "__main__":
   tracts_to_csv(argv[1], argv[2], argv[3]) 
