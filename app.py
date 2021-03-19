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

        if model == "Gradient Boosting Classifier":
            c = gbc_clf.predict(predict)
            p = gbc_clf.predict_proba(predict)
        elif model == "Random Forest Classifier":
            c = rfc_clf.predict(predict)
            p = rfc_clf.predict_proba(predict)
        elif model == "Logistic Regression":
            c = lr_clf.predict(predict)
            p = lr_clf.predict_proba(predict)
        elif model == "Support Vector Classifier":
            c = svm_clf.predict(predict)
            p = svm_clf.predict_proba(predict)
        else:
            c = None
            p = None

        if p.all():
            req_data['p_home'] = f'{p[0][0] * 100:.2f}%'
            req_data['p_away'] = f'{p[0][1] * 100:.2f}%'
        else:
            req_data['p_home'] = '--%'
            req_data['p_away'] = '--%'
        
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
        req_data['p_home'] = '--%'
        req_data['p_away'] = '--%'

    return req_data