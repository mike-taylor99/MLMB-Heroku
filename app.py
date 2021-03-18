from flask import Flask
from flask_cors import cross_origin

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'