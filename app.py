from flask import Flask, jsonify, request
from flask_cors import cross_origin
from sklearn import datasets, svm

app = Flask(__name__)

# Load Dataset from scikit-learn.
digits = datasets.load_digits()

@app.route('/')
@cross_origin()
def hello():
    return 'MLMB Backend'

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    req_data = request.get_json()
    req_data['predict_home'] = 'W'
    req_data['predict_away'] = 'L'

    return req_data