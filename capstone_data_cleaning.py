# -*- coding: utf-8 -*-


import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import plotly.io as pio
import warnings
warnings.filterwarnings("ignore")

# Need to import these your self.
# Commented out IPython magic to ensure Python compatibility.
# %run '/content/drive/MyDrive/Colab_Notebooks/Captstone_Installs_and_Packages.ipynb'
# Commented out IPython magic to ensure Python compatibility.
# %run '/content/drive/MyDrive/Colab Notebooks/PVF.ipynb'

"""# Survey Data cleaning and factor selection

## Uploading source Survey data

## Reading in xlsx

Looking at the shape of the data, we see that there are far too many columns. First i select the columns that are of particular interest and that i think might have a higher effect on wether one may or may not belive in climate change.
"""

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

surveyData = pd.read_excel('/content/drive/MyDrive/Colab_Notebooks/Untitled Folder/Drive files/CRCM_OK_May22ok_drght_listwiseML.xlsx')
surveyData.shape
rowCount = surveyData.shape[0]
colCount = surveyData.shape[1]

selectedCols = [
    'Age',
    'gend',
    'race',
    'education',
    'income',
    #------------------
    'avgtmp',
    #------------------
    'evntfreq_wind',
    'evntfreq_rain',
    'evntfreq_torn',
    'evntfreq_hail',
    #------------------
    'evntfutfreq_wind',
    'evntfutfreq_rain',
    'evntfutfreq_torn',
    'evntfutfreq_hail',
    #------------------
    #Done
    'wtr_steps_flush',
    'wtr_steps_flow',
    'wtr_steps_bath',
    'wtr_steps_lawn',
    'wtr_steps_car',
    'wtr_steps_leaks',
    'wtr_steps_ldry',
    'wtr_steps_othr',
    #------------------
    #Done
    'enrgy_sourc_elec',
    'enrgy_sourc_natgas',
    'enrgy_sourc_prop',
    'enrgy_sourc_wood',
    'enrgy_sourc_geo',
    'enrgy_sourc_othr',
    #------------------
    #Done
    'enrgy_steps_lghts',
    'enrgy_steps_ac',
    'enrgy_steps_savappl',
    'enrgy_steps_unplug',
    'enrgy_steps_insul',
    'enrgy_steps_savdoor',
    'enrgy_steps_bulbs',
    'enrgy_steps_othr',
    #------------------
    'exagrt',
    'abuse',
    #------------------
    'party',
    # Done
    'glbcc',
    'glbcc_cert', 
    'glbcc_risk',
    'glbwrm_risk_mgmt',
    'glbwrm_risk_fed_mgmt',
    #------------------
    # Global Warming policy changes
    'gcc_policy1',
    'gcc_policy2',
    'gcc_policy3',
    'gcc_policy4',
    'gcc_policy5',
    'gcc_policy6',
    'gcc_policy7',
    'gcc_policy8',
    #------------------
    'Religion_importance',
    'rel_desc',
    'Biblical_literalist',
    #------------------
    'lat',
    'lon'
]

"""We have selected 64 columns from the original 184. We can use dimenstionality reduction methods and factor colapsing to further whittle down the dataset to a more managable size. """

surveyData = surveyData[selectedCols].copy()
surveyData.shape

surveyDataNa = surveyData.apply(pd.to_numeric, errors='coerce')

na_count = surveyDataNa.isna().sum()
na_percent = na_count / len(surveyDataNa) * 100

# create a new dataframe with the results
result_df = pd.DataFrame({'Column': na_count.index, 'NA Count': na_count.values, 'NA %': na_percent.values})

# print the results
print(result_df)

"""#Data Cleaning"""

subjectInfo = [  
  'Age',
  'gend',
  'race',
  'education',
  'income'
]
df = surveyData[subjectInfo].copy()

df['income'] = pd.to_numeric(df['income'], errors='coerce')
df['gend'] = pd.to_numeric(df['gend'], errors='coerce')
df['race'] = pd.to_numeric(df['race'], errors='coerce')
df['education'] = pd.to_numeric(df['education'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')


missing_values_count = df.isna().sum()
print(missing_values_count)

del(missing_values_count)

"""## Subject Analysis

### Education

Checking to see if there are missing values in the educaiton column. As there is only a few missing i am imputing with a constant value to signify no or missing education level.
"""

df['education'].fillna(0, inplace=True)
surveyData['education'].fillna(0, inplace=True)
col = 'education'
categoryList = [0,1, 2, 3, 4, 5 , 6 ,7 ,8]
categoryListRemapped = ['Unknown or none','Less than High School', 'High School / GED', 'Vocational or Technical Training', 'Some College — NO degree', '2-year College / Associate’s Degree','Bachelor’s Degree','Master’s degree','PhD / JD (Law) / MD']
layoutTitle = 'Education Distribution'
legendTitle = 'Education'
graphTitle = "Education Distribution and Count"
create_pie_table(surveyData,col,categoryList,categoryListRemapped,layoutTitle,legendTitle,graphTitle)

"""### Age"""

fig = go.Figure()
fig.add_trace(go.Violin(y=df['Age'], box_visible=True, meanline_visible=True, points='all', jitter=0.05, name='Age Distribution'))
# # update the layout to show the annotations and provide a title and axis labels
fig.update_layout(title='Distribution of Ages', xaxis_title='Age', yaxis_title='Frequency')
fig.show()

x_col = 'Age'
title = 'Distribution of subejcts ages'
xtitle = 'Age'
ytitle = 'Number of people'
create_bar_plot(df,x_col,title,xtitle,ytitle,color = 'Blue')

# fig = go.Figure()
# fig.add_trace(go.Box(y=df['Age'], name='Age'))


age_elbow, age_lables, age_bounds = find_optimal_clusters(df['Age'], 30)

t = pd.DataFrame()
t['age_cluster'] = age_lables
t['age_range'] = t['age_cluster'].apply(lambda x: age_bounds[x])
Age_legend = t[['age_cluster','age_range']].drop_duplicates().sort_values(by=['age_cluster'], ascending=True).reset_index(drop = "True")

age_elbow.show()

def assign_age_cluster(age, age_ranges):
    for i, range in enumerate(age_ranges):
        if age >= range[0] and age <= range[1]:
            return i
    return None  



df['Age'] = df['Age'].apply(assign_age_cluster,age_ranges=Age_legend['age_range'])
surveyData['Age'] = surveyData['Age'].apply(assign_age_cluster,age_ranges=Age_legend['age_range'])




x_col = 'Age'
title = 'Distribution of subejcts ages after clustering'
xtitle = 'Age'
ytitle = 'Number of people'
create_bar_plot(df,x_col,title,xtitle,ytitle,color = 'Blue')



table_trace = go.Table(
    header=dict(values=['Cluster', 'Range']),
    cells=dict(values=[Age_legend.index, Age_legend['age_range']])
)

layout = go.Layout(title='Age Clusters mappings')

fig = go.Figure(data=[table_trace], layout=layout)
fig.show()


# del(agedf)
# del(t)
# del(Age_legend)

"""### Income"""

from sklearn.impute import KNNImputer

#Split data into two sets: one with non-missing values and another with missing values
df_train = df[df['income'].notna()]
df_test = df[df['income'].isna()]

# Fit KNN imputer on the training data
imputer = KNNImputer(n_neighbors=5)
imputer.fit(df_train[['gend', 'race', 'education', 'Age', 'income']])

# Use the trained imputer to fill missing values in the test set
imputed_data = imputer.transform(df_test[['gend', 'race', 'education', 'Age', 'income']])

df_imputed = pd.DataFrame(imputed_data, columns=['gend', 'race', 'education', 'Age', 'income'], index=df_test.index)

df_imputed.index.name = 'imputed_row_index'

df_train.index.name = 'imputed_row_index'

df_train.reset_index(inplace=True)
df_imputed.reset_index(inplace=True)


test = pd.concat([df_train, df_imputed], axis=0)
test1 = test.sort_values('imputed_row_index')

test1 = test1.set_index('imputed_row_index')

test1.reset_index(inplace=True)
test1.drop('imputed_row_index',axis = 1,inplace=True)
df = test1
# Output the rows with the imputed values
imputed_rows = df_imputed.drop(columns=['imputed_row_index'])
print('Rows with imputed values:')
print(imputed_rows)

surveyData['income'] = df['income']

Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
removed_rows = df[(df['income'] < lower_bound) | (df['income'] > upper_bound)]
# Save the removed rows to a new DataFrame
income_cap = removed_rows['income'].min()
df.loc[df['income'] > income_cap, 'income'] = income_cap
surveyData.loc[surveyData['income'] > income_cap, 'income'] = income_cap

fig = go.Figure()
fig.add_trace(go.Violin(y=df['income'], box_visible=True, meanline_visible=True, points=False, jitter=0.05, name='Income Distribution'))
# # update the layout to show the annotations and provide a title and axis labels
fig.update_layout(title='Distribution of Income', xaxis_title='Income', yaxis_title='Frequency')
fig.show()

income_elbow, income_lables, income_bounds = find_optimal_clusters(df['income'], 30)

t = pd.DataFrame()
t['income_cluster'] = income_lables
t['income_range'] = t['income_cluster'].apply(lambda x: income_bounds[x])
income_legend = t[['income_cluster','income_range']].drop_duplicates().sort_values(by=['income_cluster'], ascending=True).reset_index(drop = "True")

income_elbow.show()

def assign_income_cluster(income, income_legend):
  income_ranges = income_legend['income_range']
  for i, range in enumerate(income_ranges):
      if income >= range[0] and income <= range[1]:
          return i
  return None



df['income'] = df['income'].apply(assign_income_cluster, income_legend=income_legend)
surveyData['income'] = surveyData['income'].apply(assign_income_cluster, income_legend=income_legend)


del(t)


table_trace = go.Table(
    header=dict(values=['Cluster', 'Range']),
    cells=dict(values=[income_legend.index, income_legend['income_range']])
)

layout = go.Layout(title='Income Clusters mappings')

fig = go.Figure(data=[table_trace], layout=layout)
fig.show()

del(income_legend)

x_col = 'income'
title = 'Distribution of subejcts incomes after clustering'
xtitle = 'income'
ytitle = 'Number of people'
create_bar_plot(df,x_col,title,xtitle,ytitle,color = 'Blue')

fig = go.Figure()
fig.add_trace(go.Violin(y=df['income'], box_visible=True, meanline_visible=True, points=False, jitter=0.05, name='Income Distribution'))
# # update the layout to show the annotations and provide a title and axis labels
fig.update_layout(title='Distribution of Income', xaxis_title='Income', yaxis_title='Frequency')
fig.show()

"""### Gender"""

col = 'gend'
categoryList = [0,1]
categoryListRemapped = ['Male', 'Female']
layoutTitle = 'Gender Distribution'
legendTitle = 'Gender'
graphTitle = "Gender Distribution and Count"
create_pie_table(df,col,categoryList,categoryListRemapped,layoutTitle,legendTitle,graphTitle)

"""### Race"""

col = 'race'
race = [1, 2, 3, 4, 5 ,6,7]
raceRemapped = ['White','Black or African American','American Indian or Alaska Native','Asian','Native Hawaiian or Pacific Islander','Two or more races','Other race']
layoutTitle = 'Race Distribution'
legendTitle = 'Race'
graphTitle = "Racial Distribution and Count"
create_pie_table(surveyData,col,race,raceRemapped,layoutTitle,legendTitle,graphTitle)

"""### Mulit facetted Subject analysis

"""

labeledDf = df.copy()

race = [1, 2, 3, 4, 5 ,6,7]
raceRemapped = ['White','Black or African American','American Indian or Alaska Native','Asian','Native Hawaiian or Pacific Islander','Two or more races','Other race']
gend = [0,1]
gendRemapped = ['Male', 'Female']
edu = [0,1, 2, 3, 4, 5 , 6 ,7 ,8]
eduRemapped = ['Unknown or none','Less than High School', 'High School / GED', 'Vocational or Technical Training', 'Some College — NO degree', '2-year College / Associate’s Degree','Bachelor’s Degree','Master’s degree','PhD / JD (Law) / MD']

labeledDf['race'] = labeledDf['race'].replace(race, raceRemapped)

# Replace gender values with new labels
labeledDf['gend'] = labeledDf['gend'].replace(gend, gendRemapped)

# Replace education values with new labels
labeledDf['education'] = labeledDf['education'].replace(edu, eduRemapped)

x_axis = "Age"
y_axis ='income'
color_axis = "gend" 

titleStr = "Graph showing {0} by {1} with color showing {2}.".format(x_axis, y_axis,color_axis)

fig = px.bar(labeledDf, x=x_axis, color=color_axis,
             y=y_axis,
             title=titleStr,
             barmode='group',
             height=700,
             facet_col="education"
            )


fig.show()

"""##Perceived  global climate change risk

### How many people believe in climate change?
"""

col = 'glbcc'
categoryList = [0,1]
categoryListRemapped = ['No','Yes']
layoutTitle = 'Party Distribution'
legendTitle = 'Reponse'
graphTitle = "In your view, are-man made greenhouse gasses causing average global temperatures to rise?"
create_pie_table(surveyData,col,categoryList,categoryListRemapped,layoutTitle,legendTitle,graphTitle)

"""### Certinty and risk graphs"""

t1 = 'How certain are you that greenhouse gases are causing average global temperatures to rise?'
t2 = "How much risk do you think global warming poses for people and the environment?"
title = "Global Warming Beliefs and Risks"
create_bar_subplot(surveyData, 'glbcc_cert', 'glbcc_risk', t1, t2, title)

"""### government managment graphs"""

t1 = 'How involed should the OK Govt. be in managing the risks of Global Warming?'
t2 = 'How involed should the Fedral. be in managing the risks of Global Warming?'
title = "Government managment of global warming risks"
create_bar_subplot(surveyData, 'glbwrm_risk_mgmt', 'glbwrm_risk_fed_mgmt', t1, t2, title)



"""### Factor colapsing

#### Factor colpasing graph examples
"""

df = pd.DataFrame()
bins = [-1,0, 1, 4, 6, 8, 10]
concernlabels = ['No concern','Low concern', 'Low-medium concern', 'Medium concern', 'Medium-high concern','High concern']
risklabels = ['No risk','Low risk', 'Low-medium risk', 'Medium risk', 'Medium-high risk','High risk']

# # Factor collapse the 'values' column into bins using pd.cut()
df['glbcc_cert'] = pd.cut(surveyData['glbcc_cert'], bins=bins, labels=concernlabels)
df['glbcc_risk'] = pd.cut(surveyData['glbcc_risk'], bins=bins, labels=risklabels)

surveyData['glbcc_cert'] = pd.cut(surveyData['glbcc_cert'], bins=bins, labels=[1,2,3,4,5,6])
surveyData['glbcc_risk'] = pd.cut(surveyData['glbcc_risk'], bins=bins, labels=[1,2,3,4,5,6])

t1 = 'How certain are you that greenhouse gases are causing average global temperatures to rise?'
t2 = "How much risk do you think global warming poses for people and the environment?"
title = "Global Warming Beliefs and Risks"
create_bar_subplot(df, 'glbcc_cert', 'glbcc_risk', t1, t2, title)
del(df)

def map_values_column(column):
    def map_values(value):
        value = int(value)
        if value == -1:
            return 0
        elif value == 0: #no
            return 1
        elif value == 1 or value == 2: # low
            return 2
        elif value == 3 or value == 4: # low 
            return 3
        elif value == 5 or value == 6: # low-medium
            return 4
        elif value == 7 or value == 8: # medium
            return 5
        elif value == 9 or value == 10:# medium high 
            return 6
    return column.apply(map_values)

df = pd.DataFrame()
df['glbwrm_risk_mgmt'] = surveyData['glbwrm_risk_mgmt'].copy()
df['glbwrm_risk_fed_mgmt'] = surveyData['glbwrm_risk_fed_mgmt'].copy()


df['glbwrm_risk_mgmt'] = df['glbwrm_risk_mgmt'].fillna(-1)
df['glbwrm_risk_mgmt'] = map_values_column(df['glbwrm_risk_mgmt'])

df['glbwrm_risk_fed_mgmt'] = df['glbwrm_risk_fed_mgmt'].fillna(-1)
df['glbwrm_risk_fed_mgmt'] = map_values_column(df['glbwrm_risk_fed_mgmt'])


surveyData['glbwrm_risk_mgmt'] = df['glbwrm_risk_mgmt'].fillna(-1)
surveyData['glbwrm_risk_mgmt'] = map_values_column(df['glbwrm_risk_mgmt'])

surveyData['glbwrm_risk_fed_mgmt'] = df['glbwrm_risk_fed_mgmt'].fillna(-1)
surveyData['glbwrm_risk_fed_mgmt'] = map_values_column(df['glbwrm_risk_fed_mgmt'])

# # Define the categories beforehand
categories = ['No response','No management', 'Low management', 'Low-medium management', 'Medium management', 'Medium-high management', 'High management']
oldCategories = [0,1,2,3,4,5,6]

my_dict = dict(zip(oldCategories, categories))

df['glbwrm_risk_mgmt'] = df['glbwrm_risk_mgmt'].replace(my_dict)
df['glbwrm_risk_fed_mgmt'] = df['glbwrm_risk_fed_mgmt'].replace(my_dict)

t1 = 'How involed should the OK Govt. be in managing the risks of Global Warming?'
t2 = 'How involed should the Fedral. be in managing the risks of Global Warming?'
title = "Government managment of global warming risks"
create_bar_subplot(df, 'glbwrm_risk_mgmt', 'glbwrm_risk_fed_mgmt', t1, t2, title)

"""#### Factor colapsing inplace

## Energy conservation steps taken

### Energy Steps taken bargraph
"""

energy = ['enrgy_steps_lghts',
    'enrgy_steps_ac',
    'enrgy_steps_savappl',
    'enrgy_steps_unplug',
    'enrgy_steps_insul',
    'enrgy_steps_savdoor',
    'enrgy_steps_bulbs',
    'enrgy_steps_othr']

new_names = {
  'enrgy_steps_lghts': 'Energy efficient lights',
  'enrgy_steps_ac': 'Reduced AC usage',
  'enrgy_steps_savappl': 'Energy efficient appliances',
  'enrgy_steps_unplug': 'Unplugging Appliances',
  'enrgy_steps_insul': 'Energy saving Insulation',
  'enrgy_steps_savdoor': 'Energy saving Doors/Windows',
  'enrgy_steps_bulbs': 'Energy saving lights',
  'enrgy_steps_othr': 'Other steps'
}

df = surveyData[energy].copy()

df = df.rename(columns=new_names)

create_bin_multi_bar_plot(df, df.columns.tolist(), "Energy Saving Measures","Energy steps taken","Number of responses")

del(df)

"""### Energy steps taken dimensiontality reduction"""

if set(energy).issubset(set(surveyData.columns)):
    surveyData['EnergyStepsTaken'] = surveyData[energy].sum(axis=1)
    surveyData.drop(energy, axis=1, inplace=True)
else:
    print("One or more columns are missing in the DataFrame.")

del(energy)

"""### Bargraph   



"""

data = surveyData
x_col = 'EnergyStepsTaken'
title = "Energy saving steps taken"
xtitle = 'Number of Energy saving steps taken'
ytitle = 'Number of subjects'

create_bar_plot(data,x_col,title,xtitle,ytitle)

"""## Water Conservation Efforts taken"""

water = ['wtr_steps_flush',
    'wtr_steps_flow',
    'wtr_steps_bath',
    'wtr_steps_lawn',
    'wtr_steps_car',
    'wtr_steps_leaks',
    'wtr_steps_ldry',
    'wtr_steps_othr']

new_names = {
  'wtr_steps_flush': 'Low volume flush toilets',
  'wtr_steps_flow': 'Low volume shower head',
  'wtr_steps_bath': 'Shorter/Fewer bath/shower',
  'wtr_steps_lawn': 'Watering lawn/garden less',
  'wtr_steps_car': 'Washing car less',
  'wtr_steps_leaks': 'Repairing leaks',
  'wtr_steps_ldry': 'Less frequent laundry',
  'wtr_steps_othr': 'Other water saving steps'
}

df = surveyData[water].copy()

df = df.rename(columns=new_names)

create_bin_multi_bar_plot(df, df.columns.tolist(), "Water Saving Measures","Water saving steps taken","Number of responses")

# All the possible Water conservation steps one could take. 


if set(water).issubset(set(surveyData.columns)):
    surveyData['WaterStepsTaken'] = surveyData[water].sum(axis=1)
    surveyData.drop(water, axis=1, inplace=True)
else:
    print("One or more columns are missing in the DataFrame.")
del(water)

data = surveyData
x_col = 'WaterStepsTaken'
title = "Water saving steps taken"
xtitle = 'Number of Water saving steps taken'
ytitle = 'Number of subjects'

create_bar_plot(data,x_col,title,xtitle,ytitle)

"""## Energy Sources"""

srcs = ['enrgy_sourc_elec',
    'enrgy_sourc_natgas',
    'enrgy_sourc_prop',
    'enrgy_sourc_wood',
    'enrgy_sourc_geo',
    'enrgy_sourc_othr']

energyMapping = {
    'enrgy_sourc_elec': 'Electricity',
    'enrgy_sourc_natgas': 'Natural Gas',
    'enrgy_sourc_prop': 'Propane',
    'enrgy_sourc_wood': 'Wood',
    'enrgy_sourc_geo': 'Geothermal',
    'enrgy_sourc_othr': 'Other'
}


df = surveyData[srcs].copy()
df = df.rename(columns = energyMapping)

# First i want to get some statistical information regarding the energy sources.
EnergySourceStats = []
for i in df.columns:
  EnergySourceStats.append([str(i),str(df[i].sum()),"{:.2%}".format(df[i].sum()/rowCount)])  

EnergySourceStatHeaders = ['Energy Source Name', 'Count', 'Percentage']

df = pd.DataFrame(EnergySourceStats, columns=EnergySourceStatHeaders)
df = df.style.set_caption('Energy Source by percentage for subjects surved in Oklahoma with overlap').set_table_styles([{
    'selector': 'caption',
    'props': [
        ('color', 'White'),
        ('font-size', '15px')
    ]
}])
display(df)

# We can see that most of the energy sources tend to be Electric or Natural gas. 
# 91 percent or so are of those two types so it might make sense to just merge the remaining 9 percent
# as a new 'other' source. 

srcs2 = [
    'enrgy_sourc_prop',
    'enrgy_sourc_wood',
    'enrgy_sourc_geo',
    'enrgy_sourc_othr'
  ]

# get the first two energy sources - the 91 percent case
EnergySources = surveyData.loc[:,['enrgy_sourc_elec','enrgy_sourc_natgas']].copy()

# get the remaining energy data 
otherSourc = surveyData.loc[:,srcs2]
# There will be cases where none of the other sources are used, in a case where atleast one is used we identify that case in the 'Other_Energy_Sources' Column 
otherSourc['Other'] = otherSourc.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1)
EnergySources['Other'] = otherSourc['Other']


# Since the values in each columns are binary we can concat them all together giving a unique value that can be easily identified for every combination 
EnergySources['Energy_Sources'] = EnergySources.apply(lambda x: ''.join(x.astype(str)), axis=1)
unique_binaries = EnergySources['Energy_Sources'].unique()
# assign a unique number to each binary value
binary_to_int = {binary: i for i, binary in enumerate(unique_binaries)}
# create a new column with the encoded values
EnergySources['Energy_Sources'] = EnergySources['Energy_Sources'].map(binary_to_int)

# Now i drop the original columns and replace them with the new Energy_Sources column 
# surveyData.drop(srcs,errors = "ignore")
surveyData['EnergySources'] = EnergySources['Energy_Sources']


#Since the Energy Sources have been encoded, this table will show the mapping.
EnergySourceMapping = [
    [0, 'Electricity and Natural Gas',"{:.2%}".format(surveyData.loc[surveyData['EnergySources'] == 0].EnergySources.count()/rowCount) ],
    [1, 'Electricity',"{:.2%}".format(surveyData.loc[surveyData['EnergySources'] == 1].EnergySources.count()/rowCount)],
    [2, 'Electricty and Other Sources',"{:.2%}".format(surveyData.loc[surveyData['EnergySources'] == 2].EnergySources.count()/rowCount)],
    [3, 'Electricty, Natural Gas and Other Sources',"{:.2%}".format(surveyData.loc[surveyData['EnergySources'] == 3].EnergySources.count()/rowCount)],
    [4, 'Other Sources',"{:.2%}".format(surveyData.loc[surveyData['EnergySources'] == 4].EnergySources.count()/rowCount)],
    [5, 'Natural Gas',"{:.2%}".format(surveyData.loc[surveyData['EnergySources'] == 5].EnergySources.count()/rowCount)],
    [6, 'Natural Gas and Other Sources',"{:.2%}".format(surveyData.loc[surveyData['EnergySources'] == 6].EnergySources.count()/rowCount)],
    [7, 'Unknown or no Energy Sources',"{:.2%}".format(surveyData.loc[surveyData['EnergySources'] == 7].EnergySources.count()/rowCount)] 
]
# define the headers as a list of strings
EnergySourceHeaders = ['Source Code', 'Combination','Percentage']

df = pd.DataFrame(EnergySourceMapping, columns=EnergySourceHeaders)
df = df.style.set_caption('Energy Source by percentage for subjects surved in Oklahoma without overlap').set_table_styles([{
    'selector': 'caption',
    'props': [
        ('color', 'White'),
        ('font-size', '15px')
    ]
}])

display(df)



# Deleting the variables no longer needed. 
del(EnergySources)
del(otherSourc)
del(unique_binaries)
del(binary_to_int)
del(EnergySourceMapping)
del(EnergySourceHeaders)
del(srcs)
del(srcs2)
del(EnergySourceStats)
del(EnergySourceStatHeaders)
del(df)

"""## Weather Events"""

def create_pie_four_subplots1(df, title, labels):

    # Create the subplot
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{'type': 'pie'}] * 2, [{'type': 'pie'}] * 2],
        subplot_titles=[df.columns[0], df.columns[1], df.columns[2], df.columns[3]],
    )

    # Create the labels to use for each pie chart
    labels = labels

    # Iterate over each column of the dataframe and create a pie chart for it
    for i, col in enumerate(df.columns):
        # Count the frequency of each value in the column
        counts = df[col].value_counts()
        # Create a pie chart trace and set the name to the column name
        trace = go.Pie(values=counts.tolist(), labels=labels, name=col, textinfo='percent', hoverinfo='label+text')
        # Add the trace to the subplot
        fig.add_trace(trace, row=i // 2 + 1, col=i % 2 + 1)

    # Set the title of the subplot
    fig.update_layout(
        title={
            'text': title,
            'y': 0.99,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        # Add some space between the charts
        margin=dict(l=20, r=20, t=50, b=20),
        # Map legend labels to the new labels
        legend=dict(
            traceorder='normal',
            font=dict(
                family='sans-serif',
                size=16,
            ),

            title=dict(
                text="Responses",
                font=dict(
                    family='Arial',
                    size=20,
                    color='Black'
                )
            )
        )
    )

    fig.show()

weather = ['evntfreq_wind',
    'evntfreq_rain',
    'evntfreq_torn',
    'evntfreq_hail']

weatherMapping ={
    'evntfreq_wind' : 'Wind events last summer',
    'evntfreq_rain' : 'Rain events last summer',
    'evntfreq_torn' : 'Tronados events last summer',
    'evntfreq_hail' : 'Hail events last summer'
}
weatherCodeMapping = {
    1: 'Less frequent',
    2: 'Same frequency',
    3: 'More frequent',
}

df = surveyData[weather].rename(columns=weatherMapping)

for col in df.columns:
    # Replace column values with mapping dictionary
    df[col] = df[col].replace(weatherCodeMapping)

labels = ['Less frequent',
    'Same frequency',
    'More frequent']
    
title = "Weather events last summer"

create_pie_four_subplots(df, title, labels)

#del(df)
del(title)
del(labels)

weather = ['evntfutfreq_wind',
'evntfutfreq_rain',
'evntfutfreq_torn',
'evntfutfreq_hail']

weatherMapping ={
    'evntfutfreq_wind' : 'Wind events future summer',
    'evntfutfreq_rain' : 'Rain events future summer',
    'evntfutfreq_torn' : 'Tronados events future summer',
    'evntfutfreq_hail' : 'Hail events future summer'
}
weatherCodeMapping = {
    1: 'Less frequent',
    2: 'Same frequency',
    3: 'More frequent',
}

df = surveyData[weather].rename(columns=weatherMapping)

for col in df.columns:
    # Replace column values with mapping dictionary
    df[col] = df[col].replace(weatherCodeMapping)

labels = ['Less frequent',
    'Same frequency',
    'More frequent']
    
title = "Weather events for future summer"

create_pie_four_subplots(df, title, labels)

del(df)
del(title)
del(labels)

religion = ['Religion_importance',
    'rel_desc',
    'Biblical_literalist']

#display(surveyData[religion])

temp = ['ssn_tmp','avgtmp']

#display(surveyData[temp])

"""## Party

### Party distribution graph
"""

surveyData.loc[surveyData['party'].isnull(), 'party'] = 5
col = 'party'
categoryList = [1, 2, 3, 4, 5]
categoryListRemapped = ['Democrat', 'Republican', 'Independent', 'Other', 'Unknown']
layoutTitle = 'Party Distribution'
legendTitle = 'Party'
graphTitle = "Party Distribution and Count"
create_pie_table(surveyData,col,categoryList,categoryListRemapped,layoutTitle,legendTitle,graphTitle)

"""## Policy Distribution"""

policy = ['gcc_policy1',
    'gcc_policy2',
    'gcc_policy3',
    'gcc_policy4',
    'gcc_policy5',
    'gcc_policy6',
    'gcc_policy7',
    'gcc_policy8']

policyTitle = [
    'Set strict carbon dioxide emission limits on existing coal-fired power plant',
    'Require fossil fuel companies to pay a carbon tax',
    'Fund more research into renewable energy sources',
    'Generate renewable energy (solar and wind) on public land in the U.S.',
    'Provide tax rebates for people who purchase energy-efficient vehicles or solar panels',
    'Regulate carbon dioxide as a pollutant.',
    'Require electric utilities to produce >= 20% of their electricity costing avg 100 per household.',
    'Significantly increase reliance on nuclear energy'
]


policyMapping  = dict(zip(policy, policyTitle))

policyPossitionOld = [0,1,2,3,4]
policyPossitionNew = ['No response','Strongly support','Somewhat support', 'Somewhat oppose', 'Strongly oppose']
policydf = surveyData[policy].copy()
policydf = policydf.fillna(0)
policydf = policydf.rename(columns=policyMapping)
surveyData[policy] = surveyData[policy].fillna(0)
for col in policyTitle:
    policydf[col] = policydf[col].replace(policyPossitionOld, policyPossitionNew)

policy_dict = {policy[i]: policyTitle[i] for i in range(len(policy))}

policy_1 = policyTitle[0:4]
policy_2 = policyTitle[4:]
policydf_1 = policydf[policy_1].copy()
policydf_2 = policydf[policy_2].copy()

labels = policyPossitionNew
    
title = "Policy Responses "

# display(policydf_1)
# display(policydf_2)
create_pie_four_subplots(policydf_1, title, labels)

create_pie_four_subplots(policydf_2, title, labels)

"""## Subject opinions """

surveyData.loc[surveyData['exagrt'].isnull(), 'exagrt'] = 0
col = 'exagrt'
categoryList = [0,1, 2, 3, 4, 5]
categoryListRemapped = ['No answer/response', 'Strongly Disagree', 'Disagree', 'Nutural', 'Agree','Strongly Agree']
layoutTitle = 'Opinion'
legendTitle = 'exagrt'
graphTitle = "Has the so-called “ecological crisis” facing humankind has been greatly exaggerated."
create_pie_table(surveyData,col,categoryList,categoryListRemapped,layoutTitle,legendTitle,graphTitle)

surveyData.loc[surveyData['abuse'].isnull(), 'abuse'] = 0
col = 'abuse'
categoryList = [0,1, 2, 3, 4, 5]
categoryListRemapped = ['No answer/response', 'Strongly Disagree', 'Disagree', 'Nutural', 'Agree','Strongly Agree']
layoutTitle = 'Opinion'
legendTitle = 'abuse'
graphTitle = "Humans are severely abusing the environment."
create_pie_table(surveyData,col,categoryList,categoryListRemapped,layoutTitle,legendTitle,graphTitle)

"""## Religion"""

bins = [ -1,1, 3, 5, 7, 10]
concernlabels = [0,1,2,3,4]
surveyData['Religion_importance'] = pd.cut(surveyData['Religion_importance'], bins=bins, labels=concernlabels)

col = 'Religion_importance'
categoryList = [0,1,2,3,4]
categoryListRemapped = ['Not important at all','Not very imporant', 'indifferent', 'Important', 'Extremely Important']
layoutTitle = 'Opinion'
legendTitle = 'Religion_importance'
graphTitle = "How important is religion to you."
create_pie_table(surveyData,col,categoryList,categoryListRemapped,layoutTitle,legendTitle,graphTitle)

religion = [
  'Unknown/Not Givien',
  'Agnostic',
  'Atheist',
  'Catholic',
  'Protestant',
  'Christian (nonspecific)',
  'Jewish',
  'Mormon',
  'Other',
  'Muslim',
  'Buddhist',
  'Hindu'
]

surveyData.loc[surveyData['rel_desc'].isnull(), 'rel_desc'] = 0


col = 'rel_desc'
categoryList = [0,1,2,3,4,5,6,7,8,9,10,11]
layoutTitle = 'Opinion'
legendTitle = 'rel_desc'
graphTitle = "What religion do you follow"
create_pie_table(surveyData,col,categoryList,religion,layoutTitle,legendTitle,graphTitle)

df = surveyData.loc[surveyData['rel_desc'].isin([3,4,5,7])]
df.loc[df['Biblical_literalist'].isnull(), 'Biblical_literalist'] = -1
col = 'Biblical_literalist'
categoryList = [0,1]
categoryListRemapped = ['No','Yes']
layoutTitle = 'Opinion'
legendTitle = 'Biblical_literalist'
graphTitle = "Do you believe that the Bible is the literal word of God?"
create_pie_table(df,col,categoryList,categoryListRemapped,layoutTitle,legendTitle,graphTitle)

"""## Average Temperature"""

surveyData['avgtmp'] = surveyData['avgtmp'].fillna(0)
col = 'avgtmp'
categoryList = [0,1,2,3]
categoryListRemapped = ['No response','Decreased','No change','Increased']
layoutTitle = 'Opinion'
legendTitle = 'reponse'
graphTitle = "How would you say the average temperature has changed compared to in previous 3 years?"
create_pie_table(df,col,categoryList,categoryListRemapped,layoutTitle,legendTitle,graphTitle)

"""## Uploading and saving data"""

na_count = surveyData.isna().sum()
na_percent = na_count / len(surveyDataNa) * 100

# create a new dataframe with the results
result_df = pd.DataFrame({'Column': na_count.index, 'NA Count': na_count.values, 'NA %': na_percent.values})

# print the results
print(result_df)

surveyData.to_pickle('/content/drive/MyDrive/Colab_Notebooks/Capstone_Dataframes/surveyData_CleanedData.pickle')