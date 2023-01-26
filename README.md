# Purpose

This is a repository documenting the code I use for my final year project at UCL.

# Metrics

The metrics I will be using for my project are as follows:

// TODO

# Files

The two files necessary to complete data analsysis can be obtained from the Insideairbnb website:

http://insideairbnb.com/get-the-data/

The two files that are necessary for gathering data are:

- **reviews.csv**
- **listings_detailed.csv**

There are currently **4** source files I'm using. These are used to:

- Filter the date ranges for a given csv file from 2018 to 2022
- Recover the census tract code for a given listing using the US FCC API.
- Merge the tract codes column with the reviews.csv obtained from Inside Airbnb, which pinpoints which tract every review was left in (which is used to calculate different metrics)
- Calculate the metrics defined above (and add additional ones if necessary)
