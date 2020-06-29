# %%
import json
import requests
import os
import numpy as np
import pandas as pd

# %%

# This is the local host this app would be running on, can be changed to a real app if needed
api_url = 'http://localhost:5001/predict'

data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', 'interim', 'EV.csv'))
df = pd.read_csv(data_path)

# Make a single prediction or more than 1 and convert to json
data = df.iloc[:5, :]
data = data.values.tolist()
json_data = json.dumps(['data', data])

# %%

# Send a request to a running version of app.py and print the response predictions
headers = {"content-type": "application/json"}
json_response = requests.post(api_url, data=json_data, headers=headers)
print(json_response.text)
