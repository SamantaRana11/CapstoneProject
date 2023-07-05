#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("/Users/samantarana/Downloads/Datasets/CovidDeaths.csv")
df.head(2)


# In[3]:


X = df[["location", "date", "total_cases", "new_cases", "total_deaths", "population"]]
X.head(2)


# ## Analyzing Total cases vs Total deaths

# In[4]:


Y = df[["location", "date", "total_cases", "total_deaths"]].copy()
Y.head(2)


# In[5]:


Y["Death_Percentage"] = (Y["total_deaths"]/Y["total_cases"])*100
Y.head(2)


# In[6]:


##To get rid of warnings

#import warnings

## Filter and ignore all warnings
#warnings.filterwarnings("ignore")

## Reset the warning filters (optional)
#warnings.resetwarnings()


# In[7]:


#Choosing a random entry to ensure the calculations are applied

data = Y.iloc[82]
data


# In[8]:


#sorting Y according to location and date

sorted_Y = Y.sort_values(by=['location', 'date'], ascending=False)


# In[9]:


Y.head()


# In[10]:


data = Y.iloc[81]
data


# #### The percentage in the above dataframe, Y , shows the chances of you dying if you contract Covid in you country

# ## Analyzing Total Cases vs the Population

# In[11]:


Z = df[["location", "date", "population", "total_cases"]].copy()
Z.head(3)


# In[12]:


Z["Pop_Death_Rate"] = (Z["total_cases"]/Z["population"])*100
Z.head(2)


# In[13]:


#printing few random rows to analyze the validity of the syntax above

Z.loc[51:58,]


# ## Analyzing Countries with highest infection rates w.r.t population

# In[14]:


# Taking the maximum value out of all the total_cases as Highest Infected Count

df["HighestInfectionCount"] = df["total_cases"].max()
df.head()


# In[15]:


#Creating a subset of the dataframe df

W1 = df[["location","date", "population", "HighestInfectionCount"]].copy()
W = W1[df['continent'].notnull()].copy()


# In[16]:


#Calculating the percentage of the population who got infected country wise

W["PercentPopulationInfected"] = round((df["total_cases"].max()/df["population"])*100,2)


# In[17]:


W.head(3)


# In[18]:


import numpy as np


# In[19]:


# Grouping data so to analyze it better

grouped_data = W.groupby(['location', 'population', 'HighestInfectionCount'])

# the groupby object is not directly printed, to access the grouped data various ,methods and functions can be used,
#one of which is mean.

mean_values =np.log(grouped_data['PercentPopulationInfected'].mean())

mean_values_sorted = round(mean_values.sort_values(ascending=False),2)

print(mean_values_sorted)


# ## Observing countries with highest number of death counts per population

# In[20]:


V1 = df[["location", "total_deaths"]].copy()
V = V1[df['continent'].notnull()].copy()


# In[21]:


V["HighestDeathCount"] = V["total_deaths"].max()

# Grouping data so to analyze it better

grouped_data = V.groupby(['location'])

# the groupby object is not directly printed, to access the grouped data various ,methods and functions can be used,
#one of which is mean.

max_values = round(grouped_data['total_deaths'].max(),2)

max_values_sorted = max_values.sort_values(ascending=False)


V = pd.DataFrame(max_values_sorted)

max_values_sorted = ['TotalDeathCount']

V.columns = [max_values_sorted]

V.head()


# ## Breaking the above analyzation by continent

# In[22]:


U = df[["location", "total_deaths", "continent"]].copy()
U = U[df['continent'].notnull()]


# In[23]:


# Grouping data so to analyze it better

grouped_data = U.groupby(['continent'])

# the groupby object is not directly printed, to access the grouped data various ,methods and functions can be used,
#one of which is mean.

max_values = round(grouped_data['total_deaths'].max(),2)

max_values_sorted = max_values.sort_values(ascending=False)


U = pd.DataFrame(max_values_sorted)

max_values_sorted = ['TotalDeathCountbyContinent']

U.columns = [max_values_sorted]

U.head()


# ######Here if we observe North America and United States have the same death counts which means that Canada's death count is not included in this
#  - trying to work to be more inclusive with the numbers

# ## Possible Answer for now :-

# In[24]:


T1 = df[["location", "total_deaths"]].copy()
T = T1[df['continent'].isnull()].copy()

T = T[~T['location'].str.contains('High income')]
T = T[~T['location'].str.contains('Upper middle income')]

T["HighestDeathCount"] = T["total_deaths"].max()

# Grouping data so to analyze it better

grouped_data = T.groupby(['location'])

# the groupby object is not directly printed, to access the grouped data various ,methods and functions can be used,
#one of which is mean.

max_values = round(grouped_data['total_deaths'].max(),2)

max_values_sorted = max_values.sort_values(ascending=False)


T = pd.DataFrame(max_values_sorted)

max_values_sorted = ['TotalDeathCount']

T.columns = [max_values_sorted]

T.head()


# ## GLOBAL NUMBERS

# In[25]:


R = df[["location", "date", "new_cases", "total_deaths"]].copy()

R["total_cases"] = R["new_cases"].sum()

R["total_deaths"] = R["total_deaths"].sum()

R = R[df['continent'].notnull()].copy()

R["Death_Percentage"] = (R["total_deaths"]/R["total_cases"])*100

# Grouping data so to analyze it better

grouped_data = R.groupby(['date'])

# the groupby object is not directly printed, to access the grouped data various ,methods and functions can be used,
#one of which is mean.

sum_values = round(grouped_data['new_cases'].sum(),2)

#Aggregading per 100000of the population

sum_values = round(sum_values/100000,2)

sum_values_sorted = sum_values.sort_values(ascending=False)

R = pd.DataFrame(sum_values_sorted)

sum_values_sorted = ['Death_Percentage']

R.columns = [sum_values_sorted]

R.head()


# ### Dealing with question - 1 : Can the current vaccination rates and success rates be used to predict future trends in COVID-19 deaths globally? What are the projected impacts of successful vaccination campaigns on reducing mortality rates in the coming months

# In[26]:


df=df.fillna(value=0, axis=1)
df.head()


# In[27]:


from datetime import datetime


# In[28]:


df['date'].max()


# In[29]:


df['date'].min()


# In[30]:


df['date']=pd.to_datetime(df['date'], format='%Y-%m-%d')


# In[31]:


df['date_m']=df['date'].dt.strftime('%b-%Y')
df['date_m'].head()


# In[40]:


import plotly.express as px
import matplotlib.pyplot as plt


# In[33]:


import seaborn as sns


# In[36]:


df['moving_avg']=df.groupby(by=['location']).rolling(7).mean()['new_cases'].reset_index(drop=True)


# In[37]:


# Selecting only the required columns before performing the rolling operation
columns_to_keep = ['location', 'new_cases']
df_subset = df[columns_to_keep].copy()

# Calculating the 7-day moving average of new_cases grouped by location
df_subset['moving_avg'] = df_subset.groupby('location')['new_cases'].rolling(7).mean().reset_index(drop=True)


# In[41]:


fg = plt.figure(figsize=(16,8))
plt.title('India Covid cases by waves', fontsize=20, weight='bold')
ax=sns.lineplot(data=df[df['location']=='India'], y='moving_avg',x='date')


# In[43]:


px.choropleth(data_frame=df, locations='location',locationmode='country names',animation_frame=df['date_m'],color = 'new_cases_smoothed_per_million',range_color=(2,df['new_cases_smoothed_per_million'].max()))


# In[ ]:




