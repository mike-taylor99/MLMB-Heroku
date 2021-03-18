from flask import Flask, jsonify, request
from flask_cors import cross_origin
from sklearn import datasets, svm

app = Flask(__name__)

# Load Dataset from scikit-learn.
digits = datasets.load_digits()

@app.route('/')
@cross_origin()
def hello():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(digits.data[:-1], digits.target[:-1])
    prediction = clf.predict(digits.data[-1:])

    req_data = request.get_json()
    print(req_data)
    req_data['predict'] = repr(prediction)

    return req_data