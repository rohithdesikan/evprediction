# %%
import json
import requests
import os
import numpy as np
import pandas as pd

# %%
api_url = 'http://localhost:5001/predict'

data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', 'interim', 'EV.csv'))
df = pd.read_csv(data_path)
data = df.iloc[5, :]
data = data.values.tolist()
json_data = json.dumps(['data', data])

# %%
headers = {"content-type": "application/json"}
json_response = requests.post(api_url, data=json_data, headers=headers)
print(json_response.text)

# %%