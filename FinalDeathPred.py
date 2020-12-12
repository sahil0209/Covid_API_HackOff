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
ts_diff = pd.DataFrame(dfCaseTimeSeries.loc[:,['dateymd','totaldeceased']])


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
y = tempSeries.totaldeceased


# %%
size = int(len(X)*0.66)
X_train_xgb,X_test_xgb = X[0:size],X[size:]
y_train_xgb,y_test_xgb = y[0:size],y[size:]


# %%
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()


# %%
scaled_X_train = standard_scaler.fit_transform(X_train_xgb)
scaled_X_test = standard_scaler.transform(X_test_xgb)


# %%
from xgboost import XGBRegressor
modelXGB = XGBRegressor(n_estimators=2000,learning_rate = 0.8)
modelXGB.fit(scaled_X_train, y_train_xgb,
                 eval_set=[(scaled_X_train, y_train_xgb), (scaled_X_test, y_test_xgb)],
                 verbose=True)
modelXGB_pred = modelXGB.predict(scaled_X_test)


# %%
from sklearn.metrics import mean_absolute_error
print('XGBOOST MAE = ', (mean_absolute_error(modelXGB_pred, y_test_xgb)))


# %%
XGBOOST_df = pd.DataFrame({'y': modelXGB_pred})
XGBOOST_df.index = y_test_xgb.index


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
newPredModel = pd.DataFrame({'totaldeceased': newPred})
newPredModel.index = pd.to_datetime(latTime)


# %%
newPredModel.totaldeceased-=500


# %%
plt.figure(figsize=(12, 6))
l1, = plt.plot(tempSeries, label='Observation')
l2, = plt.plot(newPredModel, label='XGBOOST')
plt.legend(handles=[l1, l2])
# plt.savefig('XGBOOST prediction', bbox_inches='tight', transparent=False)


# %%
newPredModel


# %%
tempSeries


# %%
pickle.dump(modelXGB,open('deathPredModel.pkl','wb'))
model = pickle.load(open('deathPredModel.pkl','rb'))


# %%
believe = tempSeries
believe.reset_index()
pickle.dump(believe,open('just_check.pkl','wb'))


