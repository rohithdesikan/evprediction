# %%
# Import base packages
import os
import numpy as np
import pandas as pd 
import json
import datetime

# Use SKLearn to create train test split
from sklearn.model_selection import train_test_split

# Import AWS Sagemaker SDK to upload data
import sagemaker
import boto3

# Import local functions
from build_features import drop_nan, generate_labels

# %%
# Set up file paths
input_data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir,  'data', 'interim'))
output_data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir,  'data', 'processed'))
X_path = os.path.join(input_data_path, 'EV.csv')
y_path = os.path.join(input_data_path, 'EV_labels.csv')

# %%
# Read in files
ev = pd.read_csv(X_path)
ev_labels = pd.read_csv(y_path)

# %%
# Clean data according to build features file
ev_cleaned, ev_labels_cleaned = drop_nan(ev, ev_labels)
y_hot = generate_labels(ev_labels_cleaned)

# %%
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(ev_cleaned, y_hot, test_size = 0.2, random_state = 42)

# %%
# Save data locally to upload the entire folder contents to S3
X_train.to_csv(output_data_path + r"/X_train.csv")
pd.DataFrame(y_train).to_csv(output_data_path + r"/y_train.csv")
X_test.to_csv(output_data_path + r"/X_test.csv")
pd.DataFrame(y_test).to_csv(output_data_path + r"/y_test.csv")

# %%
# Set up sagemaker session
sagemaker_session = sagemaker.Session(default_bucket = 'rd-evprediction')

# Get default bucket
bucket = sagemaker_session.default_bucket()
print(bucket)

# Get role
role = sagemaker.get_execution_role()
print(role)

# %%
# set prefix, a descriptive name for the S3 directory
prefix = 'evpred'

# upload all data to S3
sagemaker_session.upload_data(output_data_path, bucket=bucket, key_prefix=prefix)

# %%
