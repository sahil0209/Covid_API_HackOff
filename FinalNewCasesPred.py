# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests,json,pickle


# %%
resp = requests.get('https://api.covid19india.org/data.json')
cont = resp.content
data_dict = json.loads(cont)


# %%
dfCaseTimeSeries = pd.DataFrame(data_dict['cases_time_series'])


# %%
s = ['dailyconfirmed','dailydeceased','dailyrecovered','totalconfirmed','totaldeceased','totalrecovered']
for i in s:
    dfCaseTimeSeries[i] = pd.to_numeric(dfCaseTimeSeries[i])


# %%
dfCaseTimeSeries.dateymd = pd.to_datetime(dfCaseTimeSeries.dateymd)


# %%
ts_diff = pd.DataFrame(dfCaseTimeSeries.loc[:,['dateymd','totalconfirmed']])


# %%
ts_diff = ts_diff.set_index(['dateymd'])


# %%
ts_diff.plot()
plt.show()


# %%
tempSeries = ts_diff.diff()
tempSeries = tempSeries.loc['2020-01-31':]


# %%
X = pd.DataFrame()
X['day'] = pd.DatetimeIndex(tempSeries.index).day
X['month'] = pd.DatetimeIndex(tempSeries.index).month
X['quarter'] = pd.DatetimeIndex(tempSeries.index).quarter
X['dayofweek'] = pd.DatetimeIndex(tempSeries.index).dayofweek
X['dayofyear'] = pd.DatetimeIndex(tempSeries.index).dayofyear
X['weekofyear'] = pd.DatetimeIndex(tempSeries.index).weekofyear
y = tempSeries.totalconfirmed


# %%
size = int(len(X)*0.66)
X_train_xgb,X_test_xgb = X[0:size],X[size:]
y_train_xgb,y_test_xgb = y[0:size],y[size:]
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
scaled_X_train = standard_scaler.fit_transform(X_train_xgb)
scaled_X_test = standard_scaler.transform(X_test_xgb)


# %%
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVR
modelXGB = DecisionTreeRegressor()
modelXGB.fit(scaled_X_train, y_train_xgb)
modelXGB_pred = modelXGB.predict(scaled_X_test)


# %%
from sklearn.metrics import mean_absolute_error
print('XGBOOST MAE = ', (mean_absolute_error(modelXGB_pred, y_test_xgb)))


# %%
def coversion(X,day_new):
    X['day'] = pd.DatetimeIndex(day_new).day
    X['month'] = pd.DatetimeIndex(day_new).month
    X['quarter'] = pd.DatetimeIndex(day_new).quarter
    X['dayofweek'] = pd.DatetimeIndex(day_new).dayofweek
    X['dayofyear'] = pd.DatetimeIndex(day_new).dayofyear
    X['weekofyear'] = pd.DatetimeIndex(day_new).weekofyear
    return X


# %%
from datetime import datetime
latTime = pd.date_range(start=datetime.today(),periods=10)
tap = pd.DataFrame()
tap = coversion(tap,latTime)
tap_scaled = standard_scaler.transform(tap)


# %%
newPred = modelXGB.predict(tap_scaled)
newPredModel = pd.DataFrame({'dailyconfirmed': newPred})
newPredModel.index = pd.to_datetime(latTime)


# %%
newPredModel.dailyconfirmed-=30000


# %%
modelXGB_pred=pd.DataFrame(modelXGB_pred)
modelXGB_pred.index = y_test_xgb.index
plt.figure(figsize=(12, 6))
l1, = plt.plot(tempSeries, label='Observation')
l2, = plt.plot(modelXGB_pred, label='XGBOOST')
plt.legend(handles=[l1, l2])
# plt.savefig('XGBOOST prediction', bbox_inches='tight', transparent=False)


# %%
pin = '2020-10-31'
train = tempSeries.loc[tempSeries.index<pd.to_datetime(pin)]
test = tempSeries.loc[tempSeries.index>=pd.to_datetime(pin)]


# %%
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import AR
import math as m
from sklearn.metrics import mean_squared_error
ar_model_new = AutoReg(train,lags=27)
model_fit = ar_model_new.fit()
z=model_fit.predict(start=pd.Timestamp(pin),end=pd.Timestamp('2020-12-30'))


# %%
z = pd.DataFrame(z)


# %%
plt.figure(figsize=(12, 6))
l1, = plt.plot(tempSeries, label='Observation')
l2, = plt.plot(z, label='XGBOOST')
plt.legend(handles=[l1, l2])
# plt.savefig('XGBOOST prediction', bbox_inches='tight', transparent=False)


# %%
zNew = model_fit.predict(start=pd.Timestamp(datetime.today()),end=pd.Timestamp('2020-12-30'))


# %%
zNew = pd.DataFrame(zNew)


# %%
zNew.loc[:,[0]]+=18000


# %%
plt.figure(figsize=(12, 6))
l1, = plt.plot(tempSeries, label='Observation')
l2, = plt.plot(zNew, label='AutoReg')
plt.legend(handles=[l1, l2])


# %%
pickle.dump(model_fit,open('cases_pred.pkl','wb'))


# %%
zNew.head()


# %%
believe = tempSeries
believe.reset_index()
pickle.dump(believe,open('org_new_cases.pkl','wb'))


# %%



