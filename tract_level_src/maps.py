''' file used to visualize geopandas maps for different census tracts and counties'''

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import census_data as cd


df = gpd.read_file('/Users/Kerem2001/Desktop/orleans_census_data/orleans_shape_population/acs2021_5yr_B01003_14000US22071001701.shp')

census_df = pd.read_csv('/Users/Kerem2001/Desktop/orleans_census_data/orleans_shape_population/.csv')

df['tract_code'] = df['geoid'].str.slice(7,18)

df.plot(legend=True, scheme='NaturalBreaks', column='B01003001', figsize=(10,10))

plt.title('Population by Census Tract in Orleans Parish, LA')

plt.show()

print(df.head())

print(df.info())

