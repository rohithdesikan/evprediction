# %%
import os
import numpy as np
import pandas as pd 
import flask
from flask import Flask, jsonify, request, make_response
import tensorflow as tf

from evprediction import convert_to_array

# %%
# Load saved model
# model_path = os.path.abspath(os.path.join(os.getcwd(), 'models'))
model_name = 'evmodel.h5'
model = tf.keras.models.load_model(model_name)

# %%
app = Flask(__name__)

@app.route('/') 
def hello(): 
    return "Welcome to EV Prediction"


# Works for any number of test points
@app.route('/predict', methods = ['POST'])
def make_prediction():

    # Make the request in json format
    json = request.get_json()

    # It comes in as a list of list where the 2nd element is the meter data and convert to np array
    data = json[1]
    arr = np.array(data)

    # If there is only 1 point to be tested, reshape it as necessary (1, 2880)
    if len(arr.shape) == 1:
        arr = np.reshape(arr, (-1, arr.shape[0]))

    # The House_ID could or could not be included in the data, so make sure to get rid of the 1st point
    if arr.shape[1] == 2881:
        arr =  np.array(arr[:, 1:])

    
    # Reshape array to the required shape for predictions
    arr_reshaped = np.reshape(arr, (arr.shape[0], 60, -1))

    # Use the saved model to make a prediction
    out = model.predict(arr_reshaped)

    # Reshape the output into a single dimension, convert to list and then to int (for boolean prediction)
    out_reshaped = np.reshape(out, (out.shape[0], ))
    out_pred = np.round(out_reshaped).tolist()
    out_int = [int(p) for p in out_pred]

    # Return predictions as a dictionary, works as both single and multi input prediction
    return make_response({'Predictions': out_int})

if __name__ == "__main__": 
    app.run(host ='0.0.0.0', port = 5000, debug = True)
