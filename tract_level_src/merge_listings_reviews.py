import pandas as pd
from sys import argv

# Read in the CSV files as DataFrames
def merged_csv(listings, reviews):
	# read in both csv files
	listings_df = pd.read_csv(listings, converters={'tract_code': str})
	reviews_df = pd.read_csv(reviews)
	
	# merge the two DataFrames on the 'listings_id' and 'id' columns, which correspond to each other in the different csv files
	merged_df = pd.merge(reviews_df, listings_df, left_on='listing_id', right_on='id')
	
	# add 'tract_code' column to reviews_df
	reviews_df = reviews_df.assign(tract_code = merged_df['tract_code'])
	
	# write the modified DataFrame to a new csv file
	reviews_df.to_csv('merged_reviews.csv', index=False)

if __name__ == "__main__":
	merged_csv(argv[1], argv[2])
