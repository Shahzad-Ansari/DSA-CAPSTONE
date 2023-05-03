# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
# %run '/content/drive/MyDrive/Colab_Notebooks/Captstone_Installs_and_Packages.ipynb'

import pickle
surveyData = pd.read_pickle('/content/drive/MyDrive/Colab_Notebooks/Capstone_Dataframes/surveyData_CleanedData.pickle')

"""# Geographic Information and Location Data

1st we want to drop all rows that either dont have locational data or are in a location that isnt in Oklahoma texas. To this end we use the following functions to find the State, County and FIPS codes for each of the respondants.

## Getting Geographic information
"""

fipsToCounty = {
    40001: 'Adair County',
    40003: 'Alfalfa County',
    40005: 'Atoka County',
    40007: 'Beaver County',
    40009: 'Beckham County',
    40011: 'Blaine County',
    40013: 'Bryan County',
    40015: 'Caddo County',
    40017: 'Canadian County',
    40019: 'Carter County',
    40021: 'Cherokee County',
    40023: 'Choctaw County',
    40025: 'Cimarron County',
    40027: 'Cleveland County',
    40029: 'Coal County',
    40031: 'Comanche County',
    40033: 'Cotton County',
    40035: 'Craig County',
    40037: 'Creek County',
    40039: 'Custer County',
    40041: 'Delaware County',
    40043: 'Dewey County',
    40045: 'Ellis County',
    40047: 'Garfield County',
    40049: 'Garvin County',
    40051: 'Grady County',
    40053: 'Grant County',
    40055: 'Greer County',
    40057: 'Harmon County',
    40059: 'Harper County',
    40061: 'Haskell County',
    40063: 'Hughes County',
    40065: 'Jackson County',
    40067: 'Jefferson County',
    40069: 'Johnston County',
    40071: 'Kay County',
    40073: 'Kingfisher County',
    40075: 'Kiowa County',
    40077: 'Latimer County',
    40079: 'Le Flore County',
    40081: 'Lincoln County',
    40083: 'Logan County',
    40085: 'Love County',
    40087: 'McClain County',
    40089: 'McCurtain County',
    40091: 'McIntosh County',
    40093: 'Major County',
    40095: 'Marshall County',
    40097: 'Mayes County',
    40099: 'Murray County',
    40101: 'Muskogee County',
    40103: 'Noble County',
    40105: 'Nowata County',
    40107: 'Okfuskee County',
    40109: 'Oklahoma County',
    40111: 'Okmulgee County',
    40113: 'Osage County',
    40115: 'Ottawa County',
    40117: 'Pawnee County',
    40119: 'Payne County',
    40121: 'Pittsburg County',
    40123: 'Pontotoc County',
    40125: 'Pottawatomie County',
    40127: 'Pushmataha County',
    40129: 'Roger Mills County',
    40131: 'Rogers County',
    40133: 'Seminole County',
    40135: 'Sequoyah County',
    40137: 'Stephens County',
    40139: 'Texas County',
    40141: 'Tillman County',
    40143: 'Tulsa County',
    40145: 'Wagoner County',
    40147: 'Washington County',
    40149: 'Washita County',
    40151: 'Woods County',
    40153: 'Woodward County'
}
countyToFips = {v: k for k, v in fipsToCounty.items()}

"""### Getting city, state and county

Using geocoders library i pass in the latitude and the longitude, from that i get the city state and county of each cooridnate
"""

geolocator = Nominatim(user_agent="tests",timeout=200)
def city_state_country(row):
    global counter, start_time
    counter += 1
    elapsed_time = time.time() - start_time
    if counter % 10 == 0:
      print("Time elapsed for iteration {}: {:.2f} seconds".format(counter, elapsed_time))
      print("Time per iteration: {:.2f} seconds".format(elapsed_time / counter))
    coord = f"{row['lat']}, {row['lon']}"
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    state = address.get('state', '')
    county = address.get('county','')
    row['state'] = state
    row['county'] = county
    return row

"""### Running the function

The function takes around 10-20 min to run.
"""

counter = 0
start_time = time.time()
surveyData = surveyData.apply(city_state_country, axis=1)

surveyData.to_pickle('/content/drive/MyDrive/Colab_Notebooks/Capstone_Dataframes/surveyData_CleanedData_withLocation_1.pickle')

# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

surveyData = surveyData.loc[surveyData['state'] == 'Oklahoma']

surveyData.drop(columns = ['zip','zipcode','lat',	'lon', 'city','state'],inplace = True,errors = 'ignore')

surveyData['FIPS'] = surveyData['county'].map(countyToFips)

display(surveyData)

unique_counts = surveyData["county"].value_counts()

# Convert the resulting Series to a DataFrame
unique_counts_df = unique_counts.to_frame().reset_index()

# Rename the columns
unique_counts_df.columns = ["County", "Count"]

print(unique_counts_df)

"""## SurveryData with locations

## Extracting Location data

We need to get the latititue and longitude of the counties themselves, using that we can pass those values in to get weather data for each county

We also ensure that we are only getting data for Oklahoma
"""

locations = surveyData[['county']]
locations = locations.drop_duplicates().reset_index(drop=True)

"""## Getting County Data

Since there can be multiple counties of the same name you need have both the county and state in the format 'county ,state' to get the correct data. This is doing that below
"""

counties = locations['county'].drop_duplicates().to_frame()
counties = counties.loc[counties['county'] != '']
counties['county_state'] = counties['county'] + ", Oklahoma"

"""### Using the same geocoders api i pass into

Passing in the county state information and and getting the lat and long of each county. After the function is used we delete the redundant `county_state` column
"""

from geopy.geocoders import Nominatim
# Initialize Nominatim API
geolocator = Nominatim(user_agent="MyApp")
lat = []
long = []

for county in counties['county_state']:
  location = geolocator.geocode(county)
  lat.append(location.latitude)
  long.append(location.longitude)

counties = counties.drop(['county_state'], axis=1)
counties['county_lat'] = lat
counties['county_long'] = long
counties.sort_values('county').reset_index(drop=True)

surveyData = pd.merge(surveyData, counties, on='county', how='left')

with pd.option_context("display.max_rows", None):
  display(surveyData[['county'	,'FIPS'	,'county_lat'	,'county_long']].drop_duplicates())

"""### Exporting the dataframe to xlxs"""

surveyData.to_pickle('/content/drive/MyDrive/Colab_Notebooks/Capstone_Dataframes/surveyData_with_Counties_and_FIPS.pickle')

