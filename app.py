from flask import Flask, jsonify, request
from flask_cors import cross_origin
from sklearn import datasets, svm
from joblib import dump, load
import numpy as np
import json

app = Flask(__name__)

# load models
lr_clf = load('./models/logisticreg.joblib')
svm_clf = load('./models/svm.joblib')
rfc_clf = load('./models/rfc.joblib')
gbc_clf = load('./models/gbc.joblib')

# load moving averages
with open('ma.json') as f:
  ma = json.load(f)

@app.route('/')
@cross_origin()
def hello():
    return 'MLMB Backend'

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    req_data = request.get_json()

    try:
        model = req_data['model']
        team1 = ma[req_data['team1']]
        team2 = ma[req_data['team2']]
        predict = np.array([team1 + team2])

        if model == "Neural Network":
            c = gbc_clf.predict(predict)
        elif model == "Random Forest Regressors":
            c = rfc_clf.predict(predict)
        elif model == "Linear Regression":
            c = lr_clf.predict(predict)
        elif model == "Support Vector Machine":
            c = svm_clf.predict(predict)
        else:
            c = None

        print(type(c[0]))
        if c[0] == 0:
            req_data['predict_home'] = 'W'
            req_data['predict_away'] = 'L'
        elif c[0] == 1:
            req_data['predict_home'] = 'L'
            req_data['predict_away'] = 'W'
        else:
            req_data['predict_home'] = '#'
            req_data['predict_away'] = '#'
    
    except:
        req_data['predict_home'] = '#'
        req_data['predict_away'] = '#'

    return req_data