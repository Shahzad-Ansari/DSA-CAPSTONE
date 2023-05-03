# -*- coding: utf-8 -*-


# from google.colab import files
# import pandas as pd
# uploaded = files.upload()
import json

import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.express as px
import calendar

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

# Commented out IPython magic to ensure Python compatibility.
# %run '/content/drive/MyDrive/Colab_Notebooks/Captstone_Installs_and_Packages.ipynb'

tornadoDf = pd.read_csv('/content/drive/MyDrive/Colab_Notebooks/Untitled Folder/Drive files/1950-2021_torn.csv')

surveyData =  pd.read_pickle('/content/drive/MyDrive/Colab_Notebooks/Capstone_Dataframes/surveyData_location_and_shapedata.pickle')

tornadoDf = tornadoDf.loc[tornadoDf['st'] == 'OK']
tornadoDf = tornadoDf.loc[tornadoDf['yr'].isin([2015, 2016, 2017,2018,2019,2020,2021])].reset_index()
tornadoSelectedList = ['yr', 'mo', 'dy', 'date', 'time', 'mag','f1','stf']
tornadoDf = tornadoDf[tornadoSelectedList]
col_map = {'yr': 'Year', 'mo': 'Month','dy':'Day',
           'date': 'Date','mag':'Magnitutde',
           'len':'lenght','f1':'FIPS','fc':'f-scale'
}
tornadoDf.rename(columns=col_map, inplace=True)

tornadoDf['stf'] = tornadoDf['stf'].astype(str)
pad_with_zeros = lambda x: str(x).zfill(3)
# Apply the lambda function to the 'col1' column
tornadoDf['FIPS'] = tornadoDf['FIPS'].apply(pad_with_zeros)

tornadoDf['FIPS'] = tornadoDf['stf'] + tornadoDf['FIPS']    
tornadoDf['FIPS'] = tornadoDf['FIPS'].apply(lambda x: x if x != '40000' else None)
tornadoDf = tornadoDf.drop('stf',errors='ignore',axis = 1)

tornadoDf.dropna(subset=['FIPS'], inplace=True)
tornadoDf['FIPS'] = tornadoDf['FIPS'].astype(int)
tornadoDf['county'] = tornadoDf['FIPS'].map(fipsToCounty)
tornadoDf['county'] = tornadoDf['county'].str.replace('County', '').str.strip()
mask = tornadoDf['county'].isin(surveyData['county'])

tornadoDf = tornadoDf[mask]
tornadoDf['county'] = tornadoDf['county'].str.replace('County', '').str.strip()
display(tornadoDf)

events_by_year = tornadoDf.groupby('Year').size().reset_index(name='Count')

fig = go.Figure(data=[go.Bar(x=events_by_year['Year'], y=events_by_year['Count'])])
fig.update_layout(title='Tornado of Events by Year', xaxis_title='Year', yaxis_title='Count')
fig.show()


df = tornadoDf.copy()
df['Year-Month'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
tornadoes_by_month_year = df.groupby(['Year-Month'])['Magnitutde'].count().reset_index()

fig = go.Figure()
fig.add_trace(go.Bar(x=tornadoes_by_month_year['Year-Month'], y=tornadoes_by_month_year['Magnitutde']))
fig.update_layout(title='Tornadoes by Month and Year', xaxis_title='Month and Year', yaxis_title='Number of Tornadoes')
fig.show()
fig = px.histogram(df, x='county', title='Tornado Count by county')
fig.show()

tornadoDf['Month'] = tornadoDf['Month'].apply(lambda x: calendar.month_name[x])
countiesList = ['Beaver', 'Ellis', 'Pontotoc']
tornadoDf['Year'] = tornadoDf['Year'].astype(str)

subsetDf = tornadoDf[tornadoDf['county'].isin(countiesList)]

agg_df = subsetDf.groupby(['Year', 'county','Month']).count()['Magnitutde'].reset_index()

agg_df = agg_df.rename(columns={'Magnitutde': 'count'})

x_axis = 'Year'
y_axis = 'count'
facet_col = 'county'

titleStr = 'Tornado counts by date and county'
fig = px.histogram(agg_df, x=x_axis, y=y_axis, color='Month', title=titleStr, facet_col=facet_col)

fig.update_xaxes(
    tickformat='%Y',
    title_text='Date'
)
fig.update_yaxes(
    tickmode='linear',
    dtick=1,
    title_text='Count'
)
fig.show()

