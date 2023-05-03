# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.
# %run '/content/drive/MyDrive/Colab_Notebooks/Captstone_Installs_and_Packages.ipynb'

"""# Weather Data

## Weather data API setup and parameters
"""

# Set time period
start = datetime(2015, 1, 1)
end = datetime(2021, 12, 31)



# counties = pd.read_pickle('/content/drive/MyDrive/Colab_Notebooks/Capstone_Dataframes/counties.pickle')
# fipsDF =  pd.read_pickle('/content/drive/MyDrive/Colab_Notebooks/Capstone_Dataframes/fipsDF.pickle')

locations = pd.read_pickle('/content/drive/MyDrive/Colab_Notebooks/Capstone_Dataframes/locationData.pickle')
surveyData = pd.read_pickle('/content/drive/MyDrive/Colab_Notebooks/Capstone_Dataframes/surveyData_location_and_shapedata.pickle')

"""## Get Daily  weather data"""

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

weatherDataDaily = []
counties = []
for index, row in locations.iterrows():
  location = Point(row['county_lat'],row['county_long']) # location
  daily_data = Daily(location, start, end)
  daily_data = daily_data.fetch()
  daily_data[['tavg'  ,'tmin'  ,'tmax']] = daily_data[['tavg'  ,'tmin'  ,'tmax']].applymap(celsius_to_fahrenheit)
  temp = daily_data[['tavg'  ,'tmin'  ,'tmax'  ,'prcp']]
  weatherDataDaily.append(temp)
  counties.append(row['county'])

"""## Drop rows with no weather data

There are some counties that dont have any weather data, no point in keeping them
"""

dropList_Daily  = []
for idx, x in enumerate(weatherDataDaily):
  if x.empty:
    weatherDataDaily.pop(idx)
    counties.pop(idx)

weatherByCounty = dict(zip(counties, weatherDataDaily))

def getWeatherData(countyName,my_dict):
  try:
    return my_dict[countyName]
  except KeyError:
    print('There is no weather data for this county')
    return None

getWeatherData('Cleveland',weatherByCounty)

import pickle
with open('/content/drive/MyDrive/Colab_Notebooks/Capstone_Dataframes/weatherByCounty.pickle', 'wb') as handle:
    pickle.dump(weatherByCounty, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/content/drive/MyDrive/Colab_Notebooks/Capstone_Dataframes/weatherByCounty.pickle', 'rb') as f:
    # Load the dictionary from the file
    my_dict = pickle.load(f)

getWeatherData('Cleveland',my_dict)

# Initialize a dictionary to store the key and count of NA values
na_counts = {}

# Iterate through the dataframes in the dictionary
for key, df in weatherByCounty.items():
    # Count the NA values in the dataframe and store the result in the na_counts dictionary
    na_counts[key] = df.isna().sum().sum()

print(na_counts)

daily_averages = pd.DataFrame(index=weatherByCounty[next(iter(weatherByCounty))].index)

# Iterate through the dataframes in the dictionary
for county, df in weatherByCounty.items():
    # Add the values of each day in the dataframes to the new dataframe
    daily_averages = daily_averages.add(df, fill_value=0)

# Divide the values in the new dataframe by the number of dataframes to obtain the average
daily_averages /= len(weatherByCounty)

print(daily_averages)