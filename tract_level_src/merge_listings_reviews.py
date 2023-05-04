import pandas as pd
from sys import argv

'''takes as arguments

1) census_tract_listings.csv 
2) filtered_review.csv (for counting number of reviews left in the period of study)
3) reviews.csv (unfiltered reviews file for counting number of listings added in the period of study)

and produces a new (un)filtered_merged_reviews.csv files with a new column 'tract_code' that corresponds to the census tract of the review'''

# Read in the census_tract_listings.csv and filtered reviews.csv files as DataFrames
def filtered_reviews_merged_csv(listings, reviews):
	# read in both csv files
	listings_df = pd.read_csv(listings, converters={'tract_code': str})
	reviews_df = pd.read_csv(reviews)
	
	# merge the two DataFrames on the 'listings_id' and 'id' columns, which correspond to each other in the different csv files
	merged_df = pd.merge(reviews_df, listings_df, left_on='listing_id', right_on='id')
	
	# add 'tract_code' column to reviews_df
	reviews_df = reviews_df.assign(tract_code = merged_df['tract_code'])
	
	# write the modified DataFrame to a new csv file
	reviews_df.to_csv('filtered_merged_reviews.csv', index=False)

def unfiltered_reviews_merged_csv(listings, reviews):
    # read in both csv files
	listings_df = pd.read_csv(listings, converters={'tract_code': str})
	reviews_df = pd.read_csv(reviews)

	# merge the two DataFrames on the 'listings_id' and 'id' columns, which correspond to each other in the different csv files
	merged_df = pd.merge(reviews_df, listings_df, left_on='listing_id', right_on='id')

	# add 'tract_code' column to reviews_df
	reviews_df = reviews_df.assign(tract_code = merged_df['tract_code'])

	# write the modified DataFrame to a new csv file
	reviews_df.to_csv('unfiltered_merged_reviews.csv', index=False)

if __name__ == "__main__":
	filtered_reviews_merged_csv(argv[1], argv[2])
	unfiltered_reviews_merged_csv(argv[1], argv[3])
