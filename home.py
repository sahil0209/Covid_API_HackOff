import flask
from flask import request, jsonify
import requests,datetime, json, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ar_model import AR
from datetime import datetime


app = flask.Flask(__name__)
app.config["DEBUG"] = True

recovery_pred_model_vals = pickle.load(open('recovery_cases_pred.pkl','rb'))
model = pickle.load(open('deathPredModel.pkl','rb'))
dataTemp = pickle.load(open('just_check.pkl','rb'))
cases_pred_model_vals = pickle.load(open('cases_pred.pkl','rb'))
dataNewCases = pickle.load(open('org_new_cases.pkl','rb'))
dataRecoveredCases = pickle.load(open('org_recovery_cases.pkl','rb'))

def coversion(X,day_new):
    X['day'] = pd.DatetimeIndex(day_new).day
    X['month'] = pd.DatetimeIndex(day_new).month
    X['quarter'] = pd.DatetimeIndex(day_new).quarter
    X['dayofweek'] = pd.DatetimeIndex(day_new).dayofweek
    X['dayofyear'] = pd.DatetimeIndex(day_new).dayofyear
    X['weekofyear'] = pd.DatetimeIndex(day_new).weekofyear
    return X

@app.route('/', methods=['GET'])
def home():
    return "<h1>covid19 Predictions</h1><p>This site is a prototype API for providing covid19 data and predictions.</p>"

@app.route('/api/v1/resources/predict_death', methods= ['GET'])
def predict_death():
    
    standard_scaler = StandardScaler()

    modelXGB = model

    from datetime import datetime
    latTime = pd.date_range(start=datetime.today(),periods=21)
    tap = pd.DataFrame()
    tap = coversion(tap,latTime)
    tap_scaled = standard_scaler.fit_transform(tap)

    newPred = modelXGB.predict(tap_scaled)
    newPredModel = pd.DataFrame({'totaldeceased': newPred})
    newPredModel.index = pd.to_datetime(latTime)
    newPredModel.totaldeceased-=500
    newPredModel.totaldeceased*=-1
    m = newPredModel.to_json(date_format='iso')
    
    return m

@app.route('/api/v1/resources/truth_death',methods=['GET'])
def truth_death():
    k = dataTemp.to_json(date_format ='iso')
    return k

@app.route('/api/v1/resources/predict_cases',methods=['GET'])
def predict_cases():
    zNew = cases_pred_model_vals.predict(start=pd.Timestamp(datetime.today()),end=pd.Timestamp('2020-12-30'))
    zNew = pd.DataFrame(zNew)
    zNew.loc[:,[0]]+=18000
    k =  zNew.to_json(date_format='iso')
    return k

@app.route('/api/v1/resources/truth_cases',methods=['GET'])
def truth_cases():
    k = dataNewCases.to_json(date_format='iso')
    return k

@app.route('/api/v1/resources/predict_recovery',methods=['GET'])
def predict_recovery():
    zNew = recovery_pred_model_vals.predict(start=pd.Timestamp(datetime.today()),end=pd.Timestamp('2020-12-30'))
    zNew = pd.DataFrame(zNew)
    zNew.loc[:,[0]]+=18000
    return zNew.to_json(date_format='iso')

@app.route('/api/v1/resources/truth_recovery',methods=['GET'])
def truth_recovery():
    k = dataRecoveredCases.to_json(date_format='iso')
    return k

@app.route('/api/v1/resources/past_data', methods= ['GET'])
def past_data():
    if 'date' in request.args:
        date = request.args['date']
    else:
        return "Error: No date field provided. Please specify a date."
    
    
app.run()

