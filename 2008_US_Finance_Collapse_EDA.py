#!/usr/bin/env python
# coding: utf-8

# # 2008 Economic Collapse in US(EDA)

# In[3]:


# Analysis of top US banks during financial collapse. Here i have analysed data of 13 years from 01-01-2006 to 01-01-2016.


# In[4]:


#First we need to load the data using google or yahoo API's.


# In[5]:


#importing required modules and libraries.
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

#To ignore warnings
import warnings
warnings.filterwarnings('ignore')

#for interactive graphs
import plotly
import cufflinks as cf 
cf.go_offline()

#to read data from internet
from pandas_datareader import data, wb

#for typecasting date and time
import datetime


# In[6]:


#setting start and end date for analysis
start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2019, 1, 1)


# In[7]:


#Here i am using yahoo API to read data because google API has been depricated the data.

#Bank of America(ticker = BAC)
BAC = data.DataReader('BAC', 'yahoo', start, end)

#City Group(tickre = C)
C = data.DataReader('C', 'yahoo', start, end)

#Goldman Sachs(ticker = GS)
GS = data.DataReader('GS', 'yahoo', start, end)

#JP Morgan Chase (ticker = 'JPM')
JPM = data.DataReader('JPM', 'yahoo', start, end)

#Morgan Stanley (ticker = MS)
MS = data.DataReader('MS', 'yahoo', start, end)

#Wells Fargo (ticker = WFC)
WFC = data.DataReader('WFC', 'yahoo', start, end)


# In[8]:


#Print few days of data and check the format and fields
BAC.head()

1. In above coloumns, High is the highest stock price of that day same for low and Open is opening stock price of that day.
2. Volume is nothing but number of stockes traded that day.
3. difference between Close price and Adj Close is, Close is cash or approximate closing price but Adj Close is exact closing      price after calculating divedend and other things.
# In[9]:


#Order the tickers in the list
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']


# In[10]:


#Concatenate the data of all banks in one dataframe by column
banks_data = pd.concat([BAC, C, GS, JPM, MS, WFC], axis = 1, keys = tickers)


# In[11]:


banks_data.head(2)


# In[12]:


#As you can see we need to define column names for tickers and columns of data
banks_data.columns.names = ['Bank Ticker', 'Stock Info']


# In[13]:


banks_data.head(2)


# #### Now will Start with EDA(Exploratory Data Analysis)

# In[14]:


#Lets print the Max and Min of closing price(In this case study, using 'Adj Close' price column.)
#Max
for ticker in tickers:
    print(ticker+' Closing Max - ', banks_data[ticker]['Adj Close'].max())


# In[15]:


#Another way of doing same thing by multi-indexing
banks_data.xs(key = ('Adj Close'), axis = 1, level = 'Stock Info').max()


# In[34]:


#Date on Max
banks_data.xs(key = 'Adj Close', axis = 1, level = 'Stock Info').idxmax()


# In[32]:


#Min
banks_data.xs(key = 'Adj Close', axis = 1, level = 'Stock Info').min()


# In[33]:


# Data on Min
#Min
banks_data.xs(key = 'Adj Close', axis = 1, level = 'Stock Info').idxmin()


# Now will print the percentage change in closing price after each day
# Formula = (P(i) - P(i-1))/P(i-1)
# So in python we have standard function for this.

# In[17]:


#Will store in separate data frame
returns_Of_Bank = pd.DataFrame()


# In[18]:


returns_Of_Bank = banks_data.xs(key = 'Adj Close', axis = 1, level = 'Stock Info').pct_change()


# In[19]:


returns_Of_Bank.head()


# Here in above data first row is NaN, because we can not have percent change on the very first day.

# In[20]:


#Now will visualize the returns by using histgram and scatter plot.
sns.pairplot(returns_Of_Bank[1:])


# #Now lets see the highest retuens date in the 10 years
# returns_Of_Bank.idxmax()

# In[21]:


#Now lets see the lowest retuens date in the 10 years
returns_Of_Bank.idxmin()


# As we can see that city group and JP Morgan had highest stock price on the same day i.e, 24-11-2008. But you can see that most of the companies stock's lowest price in JAN 2009. So end of 2008 and start of 2009 was financial collapse of US markets.

# In[39]:


#Now will Check whick stock is getting frequently change means risky stock.
std = returns_Of_Bank.std()


# In[45]:


std_df = pd.DataFrame(std, tickers)
std_df.columns = ['Standard Deviation']
std_df


# In[50]:


std_df.iplot()


# As you can see city group had maximum deviation, so can say it was most risky stock

# In[23]:


#Now will see that in 2018-19 how was the risk for these companies
returns_Of_Bank.ix['2018-01-01':'2019-01-01'].std()


# You can see that in 2018-19, Morgan Stanly was most risky stock.

# In[24]:


#Least risky stock
sns.distplot(returns_Of_Bank.ix['2018-01-01':'2019-01-01']['JPM'], color = 'green', bins = 50)


# In[25]:


#Highest risky Stock
sns.distplot(returns_Of_Bank.ix['2018-01-01':'2019-01-01']['MS'], color = 'green', bins = 50)


# In[26]:


#Now will visualize closing stock price in more interactive way of each companies
banks_data.xs(key = 'Adj Close', axis = 1, level = 'Stock Info').iplot()


# As you see that highest drop in stock price is in city group.

# In[27]:


#Now plotting the closing price from May 2008-09 with rolling average for 30 days
banks_data.ix['2008-05-01': '2009-05-01'].xs(key = 'Adj Close', axis = 1, level = 'Stock Info').plot(figsize = (12,6))
banks_data.ix['2008-05-01': '2009-05-01'].xs(key = 'Adj Close', axis = 1, level = 'Stock Info').rolling(window = 30).median().plot(figsize = (12,6))


# ###### Now will see the related stocks means the stocks which were lost together and gain together

# In[28]:


#Show heatmap; for that correalation matrix is required
banks_close_corr = banks_data.xs(key = 'Adj Close', axis = 1, level = 'Stock Info').corr()


# In[29]:


sns.heatmap(banks_close_corr, annot = True)


# In[30]:


#This visualisation is little complex. lets show the clustermap
sns.clustermap(banks_close_corr)


# Now you can see that WFC, JPM and GS are related and MS, BAC and C are related.

# ###### This is the end of our EDA.

# In[ ]:





# In[ ]:




