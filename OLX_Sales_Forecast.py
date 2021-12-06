#!/usr/bin/env python
# coding: utf-8

# # OLX Store Sales Forecasting 

# [Muhammad Abuzar](https://www.github.com/abuzariii)

# We have the historical sales data for 45 OLX stores located in different regions. Each store contains a number of departments, and we are tasked with predicting the department-wide sales for each store.

# ### The Dataset Structure

# We have mainly four relevant csv files:  
#     \- **stores.csv**: This file contains anonymized information about the 45 stores, indicating the type and size of store.  
#     \- **train.csv**: This is the historical training data, which covers to 2010-02-05 to 2012-11-01.  
#     \- **test.csv**: This file is identical to train.csv, except we have withheld the weekly sales. We must predict the sales for each triplet of store, department, and date in this file.  
#     \- **features.csv**: This file contains additional data related to the store, department, and regional activity for the given dates.

# The **train.csv** has the following fields:  
#     
# - Store - the store number
# - Dept - the department number
# - Date - the week
# - Weekly_Sales - sales for the given department in the given store
# - IsHoliday - whether the week is a special holiday week
# 

# The **features.csv** has the fields:  
#     
# - Store - the store number
# - Date - the week
# - Temperature - average temperature in the region
# - Fuel_Price - cost of fuel in the region
# - MarkDown1 to MarkDown5 - anonymized data related to promotional markdowns that OLX is running. MarkDown data is only available after Nov 2018, and is not available for all stores all the time. Any missing value is marked with an NA.
# - CPI - the consumer price index
# - Unemployment - the unemployment rate
# - IsHoliday - whether the week is a special holiday week
# 

# ### Importing the libraries

# In[1]:


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from datetime import datetime

from scipy.stats import probplot
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing, Holt

from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts 
from statsmodels.tsa.stattools import adfuller

sns.set_style('darkgrid')

from warnings import filterwarnings
filterwarnings('ignore')


# ### Loading the Data

# In[2]:


train = pd.read_csv('data/train.csv')
stores = pd.read_csv('data/stores.csv')
features = pd.read_csv('data/features.csv')


# In[3]:


train.head()


# In[4]:


stores.head()


# In[5]:


features.head()


# ## Data Preprocessing

# ### Checking for Null Values

# In[6]:


plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
sns.heatmap(stores.isna(), cbar = False)
plt.subplot(1,3,2)
sns.heatmap(features.isna(), cbar = False)
plt.subplot(1,3,3)
sns.heatmap(train.isna(), cbar = False)
plt.show()


# We can observe that there are null values in the **features** dataset in 7 of the columns.  
# 
# - *MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5, CPI and Unemployment*

# We can count the instances of null values in each of the columns.

# In[7]:


features.isna().sum()


# ### Combining Datasets

# We can *inner join* the **train** and **features** datasets based on *\[Store, Date]* key.

# In[8]:


temp = pd.merge(left = train, right = features ,on = ['Store', 'Date'], how = 'inner')
temp.head()


# Now we will again *inner join* this with stores.

# In[9]:


data = temp.merge(stores, on = ['Store'], how = 'inner')
data.head()


# In[10]:


data.columns


# Here now we have our entire combined data.

# In[11]:


plt.figure(figsize=(15,6))
sns.heatmap(data.isna());


# Now we will remove duplicate ***is_holiday*** column and renaming it.

# In[12]:


data.drop('IsHoliday_y', axis = 1, inplace  = True)
data = data.rename(columns = {'IsHoliday_x' : 'IsHoliday'})


# In[13]:


data.head()


# The data source mentioned the **Special Holidays** which are encorporated in ***IsHoliday*** column but not separately.  
# So we will just add theses as separate columns for analysis.

# In[14]:


data['Super_Bowl'] = np.where((data['Date'] == datetime(2010,2,10)) | (data['Date'] == datetime(2011,2,11)) | 
                               (data['Date'] == datetime(2012,2,10)) | (data['Date'] == datetime(2013,2,8)), 1, 0)
data['Labor_day'] = np.where((data['Date'] == datetime(2010,9,10)) | (data['Date'] == datetime(2011,9,9)) | 
                              (data['Date'] == datetime(2012,9,7)) | (data['Date'] == datetime(2013,9,6)), 1, 0)
data['Thanksgiving'] = np.where((data['Date']==datetime(2010, 11, 26)) | (data['Date']==datetime(2011, 11, 25)) | 
                                 (data['Date']==datetime(2012, 11, 23)) | (data['Date']==datetime(2013, 11, 29)),1,0)
data['Christmas'] = np.where((data['Date']==datetime(2010, 12, 31)) | (data['Date']==datetime(2011, 12, 30)) | 
                              (data['Date']==datetime(2012, 12, 28)) | (data['Date']==datetime(2013, 12, 27)),1,0)


# Now we can check our new data for the missing values.

# In[15]:


plt.figure(figsize=(16,6))
sns.heatmap(data.isna(), cbar = False)
plt.show()


# So after combining data corresponding to the **train** dataset we can see we are only left with the missing values in *Markdown\** columns.

# We can replace these with 0 to represent absent markdowns

# In[16]:


data = data.fillna(0)
data.isna().sum()


# In[17]:


data.describe()


# In[18]:


data[data['Weekly_Sales'] < 0]


# In[19]:


print("Number of observations with negative sales is ", len(data[data['Weekly_Sales'] < 0]))


# We will remove these observations

# In[20]:


data = data[data['Weekly_Sales'] >= 0]


# In[21]:


print('Final Shape of the Data : ',data.shape)


# ___

# We will first need to parse the dates correctly

# In[23]:


data['Date'] = pd.to_datetime(data['Date'])

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Week'] = data['Date'].dt.week
data['Day'] = data['Date'].dt.day


# ## Time Series Modeling

# #### Resampling

# We have resampled the data monthly, weekly and quarterly.  

# In[24]:


df_weekly = data.copy().set_index('Date')['Weekly_Sales']
df_weekly = df_weekly.resample('W').mean()


# In[25]:


df_monthly = data.copy().set_index('Date')['Weekly_Sales']
df_monthly = df_monthly.resample('MS').mean()


# In[26]:


df_quarterly = data.copy().set_index('Date')['Weekly_Sales']
df_quarterly = df_quarterly.resample('Q').mean()


# We'll hold some points for the testing part

# In[27]:


train_data_m = df_monthly[:int(0.7*(len(df_monthly)))]
test_data_m = df_monthly[int(0.7*(len(df_monthly))):]


# In[28]:


train_data_w = df_weekly[:int(0.7*(len(df_weekly)))]
test_data_w = df_weekly[int(0.7*(len(df_weekly))):]


# In[29]:


train_data_q = df_quarterly[:int(0.7*(len(df_quarterly)))]
test_data_q = df_quarterly[int(0.7*(len(df_quarterly))):]


# In[30]:


plt.figure(figsize=(12,5))
train_data_m.plot()
plt.title('Monthly Sampled Data')
plt.show()


# In[31]:


plt.figure(figsize=(12,5))
train_data_w.plot()
plt.title('Weekly Sampled Data')
plt.show()


# In[32]:


plt.figure(figsize=(12,5))
train_data_q.plot()
plt.title('Quarterly Sampled Data')
plt.show()


# In[33]:


from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
rcParams['figure.figsize'] = 16,18
seasonal_decompose(df_weekly, model = 'additive').plot()
plt.show()


# - The trend of weekly sales is such that it first decreases then elevates.
# - Given OLX *dataset is **seasonal** dataset* as it is repeating the same pattern in the month of Nov-Dec.

# ______

# Now we may require to perform some tests recurringly so we will define some reusable functions.

# ### Log Likelihood Ratio Test

# For Model comparisions we will define a function for **Log Likelihood Ratio Test**

# In[34]:


def LLR_test(mod_1, mod_2, DF=1):
    '''
    input model_1, model_2, degree of freedom (default = 1)
    '''
    L1 = mod_1.llf
    L2 = mod_2.llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR, DF).round(3)
    return p


# ### Augmented Dickey-Fuller Test

# We will define a Dickey-Fuller Test function for testing stationarity.

# In[35]:


def adickeyFuller(series):
    result = adfuller(series)

    print('---Results of Dickey Fuller----\n')
    print('ADF Statistic: ', result[0])
    print('p-value: ', result[1])
    print('Critical Values:')
    for key,value in result[4].items():
        print('{}: {}'.format(key,value))


# ### ACF and PACF Plots

# We will define a combined for plotting acfs and pacfs

# In[36]:


def acf_pacf( series, acf = False, pacf = False, lags_ = None, title = ''):
    rcParams['figure.figsize'] = 8,3    
    if acf == True:
        sgt.plot_acf(series, zero = False, lags=lags_)
        plt.title("ACF "+ title + " ", size=20)
        plt.show()
    if pacf == True:
        sgt.plot_pacf(series, zero = False, lags = lags_)
        plt.title("PACF "+ title + " ", size=20)
        plt.show()


# ## ----------------------------------------------- FOR WEEKLY SAMPLING --------------------------------------------

# #### Checking Stationarity

# In[37]:


print("On Weekly Sampled Data\n")
print(adickeyFuller(df_weekly))


# - Since the p-value of the test is much less than the significance values for even 1%,   
# we can be sure that our series is Stationary.

# ## The ***A***uto ***R***egression Model

# We generally consider PACF for Auto Regressions.

# In[38]:


acf_pacf(train_data_w, pacf= True, title = 'for Weekly Sampling ')


# - The above plots suggests that we have 3-4 significant values so we can try models having 3 lags to atleast 5 lags.

# #### AR(1)

# In[39]:


model_w_ar_1 = ARMA(train_data_w, order= (1,0)).fit()
model_w_ar_1.summary()


# - Since the p-values are significant for both the constant and the coefficient, we can try higher lag models.

# #### AR(3)

# In[40]:


model_w_ar_3 = ARMA(train_data_w, order= (3,0)).fit()
model_w_ar_3.summary()


# #### AR(5)

# In[41]:


model_w_ar_5 = ARMA(train_data_w, order= (5,0)).fit()
model_w_ar_5.summary()


# - We see that we have 4 significant values and 3 non-significant.
# - Also our latest lag is non-significant, so we should be stopping here.
# 
# Although we can have a look at how another lag higher model would perform.

# #### AR(6)

# In[42]:


model_w_ar_6 = ARMA(train_data_w, order= (6,0)).fit()
model_w_ar_6.summary()


# - As expected we can observe that our last lag is non-significant so AR(6) is probably a reach..  

# Let us see what the ***LLR Test*** suggest

# In[43]:


print('\nLLR For AR(1) and AR(3):\n',LLR_test(model_w_ar_1, model_w_ar_3))
print('\nLLR For AR(3) and AR(5):\n',LLR_test(model_w_ar_3, model_w_ar_5))
print('\nLLR For AR(5) and AR(6):\n',LLR_test(model_w_ar_5, model_w_ar_6))


# - The above test suggests that more complex model than AR(3) is significantly better, 
# - Although it also suggests that using AR(6) orver AR(5) would not be better for the model. 
# 
# But we will consider the results of AR(6) for comparisions.

# In[44]:


plt.figure(figsize=(20,15))

plt.subplot(3,1,1)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('AR(3)', size = 18)
plt.plot(model_w_ar_3.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.subplot(3,1,2)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('AR(5)', size = 18)
plt.plot(model_w_ar_5.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.subplot(3,1,3)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('AR(6)', size = 18)
plt.plot(model_w_ar_6.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.show()


# #### Observing the residuals

# In[45]:


print('Stationarity for Resiiduals\n')
adickeyFuller(model_w_ar_5.resid)


# The residual series is stationary but it is not same as white noise.   
# We should also consider the ACF plots and none of the points should be significant.

# In[46]:


acf_pacf(model_w_ar_5.resid, pacf = True, title = 'for AR(5) Residuals')


# - We can see that there are no significant points

# In[47]:


model_w_ar_5.resid.plot(figsize= (20,5))
plt.title('Residual of Weekly Sales', size = 18)
plt.show()


# - Ideally the residuals should signify a ***Random Walk*** but,  
# there seems to be an unaccounted pattern at the end of the year in the residuals.   
# 
# So we should try other models as well.

# ## The ***M***oving ***A***verage Model

# For Moving Averages we consider the ACF Plot.

# In[48]:


acf_pacf(train_data_w, acf= True, title = "for Weekly Sampling")


# #### MA(1)

# In[49]:


model_w_ma_1 = ARMA(train_data_w, order= (0,1)).fit()
model_w_ma_1.summary()


# #### MA(3)

# In[50]:


model_w_ma_3 = ARMA(train_data_w, order= (0,3)).fit()
model_w_ma_3.summary()


# #### MA(4)

# In[51]:


model_w_ma_4 = ARMA(train_data_w, order= (0,4)).fit()
model_w_ma_4.summary()


# #### MA(5)

# In[52]:


model_w_ma_5 = ARMA(train_data_w, order= (0,5)).fit()
model_w_ma_5.summary()


# - Our lag is not significant anymore, so we will stop here.
# 
# And Compare the above models' performances

# ***LLR Test*** Results.

# In[53]:


print('\nLLR For MA(1) and MA(3):\n',LLR_test(model_w_ma_1, model_w_ma_3))
print('\nLLR For MA(3) and MA(4):\n',LLR_test(model_w_ma_3, model_w_ma_5))
print('\nLLR For MA(4) and MA(5):\n',LLR_test(model_w_ma_4, model_w_ma_5))


# So using using MA(5) is not significant as shown in the test as well.

# In[54]:


plt.figure(figsize=(20,15))

plt.subplot(3,1,1)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('MA(3)', size = 18)
plt.plot(model_w_ma_3.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.subplot(3,1,2)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('MA(4)', size = 18)
plt.plot(model_w_ma_4.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.subplot(3,1,3)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('MA(5)', size = 18)
plt.plot(model_w_ma_5.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.show()


# In[55]:


acf_pacf(model_w_ma_5.resid, acf = True, title = 'for MA(5) Residuals')


# ## The ***A***uto ***R***egression ***M***oving ***A***verage Models

# In[56]:


acf_pacf(train_data_w, acf = True, pacf= True)


# For ARMA we will begin with complex models and slowly step down.

# #### ARMA(5,2)

# In[57]:


model_w_arma_5_2 = ARMA(train_data_w, order= (5,2)).fit()
model_w_arma_5_2.summary()


# #### ARMA(5,1)

# In[58]:


model_w_arma_5_1 = ARMA(train_data_w, order= (5,1)).fit()
model_w_arma_5_1.summary()


# #### ARMA(3,1)

# In[59]:


model_w_arma_3_1 = ARMA(train_data_w, order= (3,1)).fit()
model_w_arma_3_1.summary()


# In[60]:


print('\nLLR For ARMA(3,1) and ARMA(5,1):\n',LLR_test(model_w_arma_3_1, model_w_arma_5_1))
print('\nLLR For ARMA(5,1) and ARMA(5,2):\n',LLR_test(model_w_arma_5_1, model_w_arma_5_2))


# - ARMA(5,1) is significantly better than ARMA(3,1)  
# - ARMA(5,2) isn't better than ARMA(5,1)

# <span style="color:darkred">Note: If any of the nesting conditions had been broken, our LLR test would have been become void.</span>

# In[61]:


plt.figure(figsize=(20,15))

plt.subplot(3,1,1)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('ARMA(3,1)', size = 18)
plt.plot(model_w_arma_3_1.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.subplot(3,1,2)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('ARMA(5,1)', size = 18)
plt.plot(model_w_arma_5_1.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.subplot(3,1,3)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('ARMA(5,2)', size = 18)
plt.plot(model_w_arma_5_2.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.show()


# #### Plots with probabilistic predictions

# In[62]:


rcParams['figure.figsize'] = 20,5
model_w_arma_3_1.plot_predict(end= '2012-10-28')
model_w_arma_5_1.plot_predict(end= '2012-10-28')
model_w_arma_5_2.plot_predict(end= '2012-10-28')
plt.show()


# In[63]:


model_w_arma_5_1.resid.plot(figsize= (20,5))
plt.title('Residual of Weekly Sales', size = 18)
plt.show()


# The residual pattern is still similar which was bound to happen.

# ## The ***A***uto ***R***egression ***I***ntegrated ***M***oving ***A***verage Model

# #### <span style="color:darkred">Note: Since our Series is already stationary, we better not use ARIMA as we don't need the differencing part. I am  just trying  ARIMA here for exploratory purpose.</span>

# In[64]:


model_w_arima_1_1_1 = ARIMA(train_data_w, order= (1,1,1)).fit()
model_w_arima_5_1_1 = ARIMA(train_data_w, order= (5,1,1)).fit()
model_w_arima_5_1_2 = ARIMA(train_data_w, order= (5,1,2)).fit()


# In[65]:


print("ARIMA(1,1,1):  \t LL = ", model_w_arima_1_1_1.llf, "\t AIC = ", model_w_arima_1_1_1.aic)
print("ARIMA(5,1,1):  \t LL = ", model_w_arima_5_1_1.llf, "\t AIC = ", model_w_arima_5_1_1.aic)
print("ARIMA(5,1,2):  \t LL = ", model_w_arima_5_1_2.llf, "\t AIC = ", model_w_arima_5_1_2.aic)


# The ***ARIMA (5, 1, 1)*** seems to be a decent fit among these three.

# In[66]:


print('\nLLR For ARIMA(1,1,1) and ARIMA(5,1,1):\n',LLR_test(model_w_arima_1_1_1, model_w_arima_5_1_1))
print('\nLLR For ARIMA(5,1,1) and ARIMA(5,1,2):\n',LLR_test(model_w_arima_5_1_1, model_w_arima_5_1_2))
# print('\nLLR For AR(5) and AR(6):\n',LLR_test(model_w_ar_5, model_w_ar_6))


# In[67]:


plt.figure(figsize=(20,15))

plt.subplot(3,1,1)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('ARIMA(1,1,1)', size = 18)
plt.plot(model_w_arima_1_1_1.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.subplot(3,1,2)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('ARIMA(5,1,1)', size = 18)
plt.plot(model_w_arima_5_1_1.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.subplot(3,1,3)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('ARIMA(5,1,2)', size = 18)
plt.plot(model_w_arima_5_1_2.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.show()


# - As expected we are using differencing which is not required in our stationary series which is giving us these vertically shifted results.  

# ________________

# #### None of our models that we have created address the Seasonality in the series.   
# #### So now we will now create models that support seasonality.

# ## The ***S***easonal-***ARMA*** Model

# #### Format for SARIMAX (***p***, ***d***, ***q***) (***P***, ***D***, ***Q***, ***s***)  
# 
# **p** - Trend Auto Regression  
# **d** - Trend Difference Order  
# **q** - Trend Moving Average  
# 
# **P** - Seasonal Auto Regression  
# **D** - Seasonal Difference Order  
# **Q** - Seasonal Moving Average
# 
# **s** - Length of Cycle

# Here the exogenous parameter will be dropped to use *SARIMAX* as *SARIMA* and we'll also keep the differencing order 0 as our series is stationary.  
# So we will have *SARMA* as final result.

# In[68]:


plt.title('Weekly Sales for 2010')
plt.plot(df_weekly[:48])
plt.show()


# Since the hike appears after every 1 year and our periods are in weeks, we should be considering ***52*** as the ***length of cycle(s)***.

# In[69]:


model_w_sarma_101_101_52 = SARIMAX(train_data_w, order = (1,1,1), seasonal_order= (1,0,1,52)).fit()
model_w_sarma_500_100_52 = SARIMAX(train_data_w, order = (5,0,0), seasonal_order= (1,0,0,52)).fit()
model_w_sarma_501_101_52 = SARIMAX(train_data_w, order = (5,0,1), seasonal_order= (1,0,1,52)).fit()


# In[70]:


plt.figure(figsize=(20,15))

plt.subplot(3,1,1)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('SARMA(1,1,1)(1,0,1,52)', size = 18)
plt.plot(model_w_sarma_101_101_52.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.subplot(3,1,2)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('SARMA(5,0,0)(1,0,0,52)', size = 18)
plt.plot(model_w_sarma_500_100_52.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.subplot(3,1,3)
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('SARMA(5,0,1)(1,0,1,52)', size = 18)
plt.plot(model_w_sarma_501_101_52.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.legend()

plt.show()


# #### Observing Error Terms

# In[71]:


acf_pacf(model_w_sarma_501_101_52.resid, acf= True, pacf= True, title = 'for SARMA (5,0,1) (1,0,1,52) Residuals')


# In[72]:


plt.suptitle('Comparision of Residual Plots', size = 16)

model_w_arma_5_1.resid.plot(figsize= (20,5), c = 'tab:green', label = 'ARMA(5,1)')
plt.title('Residual of Weekly Sales For ARMA(5,1)', size = 14)
plt.legend()
plt.show()
model_w_sarma_500_100_52.resid.plot(figsize= (20,5), c= 'tab:green', label = 'SARMA (5,0,0) (1,0,0,52)')
plt.title('Residual of Weekly Sales For SARMA', size = 14)
plt.legend()
plt.show()


# - As evident from the residual plots of both of our higher lag **SARMA models**, we can now see that some of the unaccounted pattern that was emerging due to seasonality earlier has been accounted.  
# - We can still see a **sharp variation** once which could be due to the inconsistency that for year **2010** our data starts from February.  

# In[73]:


# DONT RUN THIS

# plt.suptitle('ARIMA Forecast From Monthly, Weekly and Quarterly Sampling', size = 18)

# for i, mwq in enumerate(['m', 'w', 'q']):
    
#     model_tmp = eval('arima_'+str(mwq)+'_fit')
#     test_data_tmp = eval('test_data_'+str(mwq))
#     train_data_tmp = eval('train_data_'+str(mwq))
    
#     tmp = pd.DataFrame(model_tmp.forecast(len(test_data_tmp))[0])
#     tmp.index = test_data_tmp.index

#     plt.subplot(3,1,i+1)
#     plt.plot(tmp, c = 'tab:orange', label = 'Predicted Sales')
#     plt.plot(test_data_tmp, c= 'tab:green', label = 'Actual Sales')
#     plt.plot(train_data_tmp, c= 'tab:green')
#     plt.legend()
#     plt.show()
    


# ## Smoothing

# In[74]:


model_w_es_m = ExponentialSmoothing(train_data_w, seasonal_periods= 52, trend= 'multiplicative', seasonal= 'multiplicative').fit()
model_w_se_m = SimpleExpSmoothing(train_data_w).fit(smoothing_level=0.5, optimized=False)
model_w_hw_m = Holt(train_data_w).fit()


# In[75]:


# sns.lineplot(df_weekly.index[101:], model_se_m.predict(43))


# In[76]:


plt.figure(figsize=(20,5))
plt.plot(train_data_w, c= 'tab:blue')
plt.plot(test_data_w, c= 'tab:blue')
plt.title('Smoothing', size = 18)
plt.plot(model_w_es_m.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Exponential Smoothing')
# plt.plot(model_w_se_m.predict(0), c= 'tab:orange', label = 'Simple Smoothing')
plt.plot(model_w_es_m.predict(0),c= 'tab:green', label = 'Exponential Smoothing')
# plt.plot(model_w_se_m.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Exponential Smoothing')
plt.legend()
plt.show()


# ## Regression Model

# In[77]:


X = data.copy()


# In[78]:


# X = data[data['Dept'] == 5]
# X = X[X['Store'] == 5]


# #### Label Encoding

# In[79]:


X_full = X[['Store', 'IsHoliday', 'Temperature', 'Dept',
       'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
       'MarkDown5', 'CPI', 'Unemployment', 'Type', 'Size', 'Year', 'Month', 'Week','Day']]

y_full = X['Weekly_Sales']


# In[80]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[81]:


X_full['Type'] = encoder.fit_transform(X_full['Type'])
X_full['IsHoliday'] = encoder.fit_transform(X_full['IsHoliday'])


# #### Choosing the most important features.

# In[82]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(max_depth= 10)
regressor.fit(X_full, y_full)


# In[83]:


importances = regressor.feature_importances_
idx = importances.argsort()[-15:]


# In[84]:


plt.figure(figsize=(15,4))
fig = sns.barplot(x = X_full.T.iloc[idx].index, y = importances[idx])
fig.set_xticklabels(fig.get_xticklabels(), rotation = 90)
plt.show()


# Selecting the important features.

# In[85]:


X_full = X_full[['Dept','Size', 'Store', 'Month', 'Type', 'CPI', 'Temperature', 'Store', 'Fuel_Price','Day', 'MarkDown3']]


# In[86]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)


# #### Splitting the dataset

# In[87]:


from sklearn.model_selection import train_test_split


# In[88]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_full, test_size = 0.3, random_state = 101)


# In[89]:


# X_train = X_full[:100]
# X_test = X_full[100:]
# y_train = y_full[:100]
# y_test = y_full[100:]


# In[90]:


# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# l_regressor = LinearRegression()
# l_regressor.fit(X_scaled, y_train)

rf_regressor = RandomForestRegressor()
rf_regressor.fit(X_train, y_train)


# In[91]:


pred = rf_regressor.predict(X_train)
pred_test = rf_regressor.predict(X_test)


# In[92]:


from sklearn.metrics import r2_score
print("The Adjusted-R2 Score is", r2_score(y_test, pred_test))


# In[93]:


plt.figure(figsize=(20,5))
plt.plot(list(y_test.reset_index()['Weekly_Sales'][::1000]),alpha = 0.8, label = 'Actual Values', c= 'r')
plt.plot(pred_test[::1000],alpha = 0.8,c = 'b', label = 'Predictions')
plt.title('Random Forest Regression Results', size = 14)
plt.legend()
plt.show()


# In[94]:


# tmp = pd.DataFrame(y_train)
# tmp.index = X['Date'][:100]
# tmp_test = pd.DataFrame(y_test)
# tmp_test.index = X['Date'][100:]


# tmp_pred = pd.DataFrame(pred)
# tmp_pred.index = X['Date'][:100]
# tmp_test_pred = pd.DataFrame(pred_test)
# tmp_test_pred.index = X['Date'][100:]


# In[95]:


# plt.figure(figsize=(20,5))
# plt.plot(tmp, label = 'Train Set Actual')
# plt.plot(tmp_pred, label = 'Train Set Predictions')

# plt.plot(tmp_test, label = 'Test Set Actual')
# plt.plot(tmp_test_pred, label = 'Test Set Predictions')

# plt.title('Linear Regression')
# plt.legend()
# plt.show()


# ## Facebook **PROPHET**

# In[96]:


from fbprophet import Prophet


# In[97]:


datetime.strptime('2012-01-01','%Y-%m-%d')
php = Prophet(daily_seasonality=False,weekly_seasonality=False,yearly_seasonality=True)

tmp = train_data_w.reset_index().rename(columns={'Date':'ds','Weekly_Sales':'y'})
php.fit(tmp)


# In[98]:


future = php.make_future_dataframe(periods = 43+52, freq= 'W')
forecast = php.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[99]:


php.plot(forecast, figsize=(16,5))
plt.plot(test_data_w,'k', label = 'Test Actual Values', alpha = 0.8)
plt.vlines(datetime.strptime('2012-01-01','%Y-%m-%d'), ymin=10000, ymax = 27500, alpha = 0.5, color= 'tab:orange')
plt.vlines(datetime.strptime('2012-10-28','%Y-%m-%d'), ymin=10000, ymax = 27500, alpha = 0.5, color= 'g')
plt.title('Prophet', size = 18)
plt.legend()
plt.show()


# In[100]:


php.plot_components(forecast)
plt.show()


# **<span style=color:darkred>Note: Dyanmic plots may not work on some platforms</style></span>**

# In[101]:


from fbprophet.plot import plot_plotly
import plotly.offline as py
# import plotly.graph_objects as go
py.init_notebook_mode()

fig = plot_plotly(php, forecast)
py.iplot(fig)


# ## ---------------------------------------------------------------------------------------------------------------

# ## Forecast on Single Dept and Single Store

# Since all the studies tilll now were done on the resampled dataset of all departments aggregated over mean, the series did not belong to any particular department or any store.  

# Each department would be a unique Time-Series on it's own as we can see from the plot below which is only for 10 departments out of all 81

# In[102]:


plt.figure(figsize=(20,5))
for i in range(1,10):
    tmp = data[data['Dept'] == i].set_index('Date').resample('W').mean()
    tmp_x = tmp.reset_index()['Date']
    tmp_y = tmp.reset_index()['Weekly_Sales']
    sns.lineplot(tmp_x, tmp_y, label = 'Dept_'+str(i))
plt.show()


# We will now choose a particular department and average out its values across the stores.  
# And we will access how they are previous models perform on individual departments.

# In[103]:


X_f = data[data['Dept'] == 2]
X_f = X_f.set_index('Date').resample('W').mean() 


# In[104]:


train_lr = X_f['Weekly_Sales'][:len(train_data_w)]
test_lr = X_f['Weekly_Sales'][len(train_data_w):]


# In[105]:


tmp = pd.DataFrame(train_lr)
# tmp.index = X_f['Date'][:len(train_data_w)]
# ddf = ddf.resample('W').mean()

tmp_test = pd.DataFrame(test_lr)
# tmp_test.index = X_f['Date'][len(train_data_w):]
# ddf_test = ddf_test.resample('W').mean()


# In[106]:


plt.figure(figsize=(20,5))
plt.plot(tmp, label = 'train Set Values')
plt.plot(tmp_test, label = 'Test Set Values')
plt.title('Complete Actual Series', size = 18)
plt.legend()
plt.show()


# In[107]:


# SARIMAX(tmp, order = (3,0,3), seasonal_order= (1,0,0,52)).fit().summary()


# For the forecasting model, we will include the observations from the test.

# ### SARMA Performance

# In[108]:


model_r_sarma_303_100_52 = SARIMAX(tmp, order = (3,0,2), seasonal_order= (1,0,0,52)).fit()


# In[109]:


# model_r_sarma_303_100_52.summary()


# In[110]:


plt.figure(figsize=(20,7))
plt.plot(tmp, c= 'tab:blue')
plt.plot(tmp_test, c= 'tab:blue', alpha = 1)
plt.title('SARMA(3,0,3)(1,0,0,52)', size = 18)
plt.plot(model_r_sarma_303_100_52.predict(start='2012-01-08', end= '2012-10-28'), c= 'tab:green', label = 'Prediction')
plt.plot(model_r_sarma_303_100_52.predict(start='2012-10-28', end= '2013-12-15'), c= 'tab:orange', label = 'Forecast')
# plt.yticks(range(0,15000,5000))
plt.legend()
plt.show()


# ### Prophet Performance

# In[111]:


tmp = tmp.reset_index().rename(columns = {'Date':'ds', 'Weekly_Sales':'y'})


# In[112]:


m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality= True)
m.fit(tmp)


# In[113]:


future = m.make_future_dataframe(periods=43+52, freq= 'W')
forecast = m.predict(future)
fig1 = m.plot(forecast, figsize=(16,5))
plt.plot(tmp_test, c= 'k', alpha = 0.8, label = 'Test Actual Values')
plt.vlines(datetime.strptime('2012-01-01','%Y-%m-%d'), ymin=30000, ymax = 60000, alpha = 0.5, color= 'tab:orange')
plt.vlines(datetime.strptime('2012-10-28','%Y-%m-%d'), ymin=30000, ymax = 60000, alpha = 0.5, color= 'g')
plt.title('Prophet', size = 18)
plt.legend()
plt.show()


# **<span style=color:darkred>Note: Dyanmic plots may not work on some platforms</style></span>**

# In[114]:


from fbprophet.plot import plot_plotly
import plotly.offline as py
# import plotly.graph_objects as go
py.init_notebook_mode()

fig = plot_plotly(m, forecast)
py.iplot(fig)


# ### and that's it!
