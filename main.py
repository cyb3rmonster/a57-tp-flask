import pickle
from flask import Flask, request, render_template
import numpy as np
import sklearn
import pandas as pd
import boto3
from botocore.client import Config
import os

s3 = boto3.resource('s3',
                    endpoint_url='https://minio.cyber.monster:9000',
                    aws_access_key_id='tp-hafed',
                    aws_secret_access_key='strong-password',
                    config=Config(signature_version='s3v4'),
                    region_name='us-east-1')

app=Flask(__name__)

xx = [10.1,0.37, 0.34, 2.4, 0.085, 5.0, 17.0, 0.99683, 3.17, 0.65, 10.6]
# xy = np.array(xx).reshape(1, -1)
# predicted1 = lr.predict(xy)
# print(predicted1)

def ValuePredictor(to_predict_list):
    names = ['fixed acidity' ,'volatile acidity' ,'citric acid' ,'residual sugar' ,'chlorides' ,'free sulfur dioxide' ,'total sulfur dioxide' ,'density'   ,'pH' ,'sulphates' ,'alcohol']
    to_predict = np.array(to_predict_list).reshape(1, -1)
    to_predict1 = pd.DataFrame(to_predict,columns=names)
    loaded_model = pickle.load(open("model.dat", "rb"))
    result = loaded_model.predict(to_predict1)
    return result[0]
 
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        s3.Bucket('tp-a57').download_file('latestmodel/model.dat', 'model.dat')
        prediction = ValuePredictor(request.json)                 
        return str(prediction)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
# y = ValuePredictor(xx)
# print(y)