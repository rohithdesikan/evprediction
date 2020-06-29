# %%
import os
import numpy as np
import pandas as pd 
import flask
from flask import Flask, jsonify, request
import tensorflow as tf

from evprediction.models.predict import convert_to_array

# %%
# Load saved model
model_path = os.path.join(os.getcwd(), 'models')
model_name = 'evmodelV2.h5'
model = tf.keras.models.load_model(model_path + r'/' + model_name)

# %%
app = Flask(__name__)

@app.route('/') 
def hello(): 
    return "welcome to EV Prediction"


@app.route('/', methods = ['POST']) 
def make_prediction():
    json = request.get_json()
    df = pd.DataFrame(json)
    arr = df.values
    out = model.predict(arr)

    return out
  
  
# %%
if __name__ == "__main__": 
    app.run(host ='0.0.0.0', port = 5001, debug = True)  
