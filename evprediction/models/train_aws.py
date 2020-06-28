# %%
# Import base packages
import os
import numpy as np

# Import AWS training package
import sagemaker
from sagemaker.tensorflow import TensorFlow
import boto3

# %%
# Set up sagemaker session
sagemaker_session = sagemaker.Session(default_bucket = 'rdevprediction')

# Get default bucket
bucket_name = sagemaker_session.default_bucket()
print(bucket_name)

# Get role
role = sagemaker.get_execution_role()
print(role)

# set prefix, a descriptive name for the S3 directory
prefix = 'evpred'

train_dir = f's3://{bucket_name}/{prefix}/train/'
test_dir = f's3://{bucket_name}/{prefix}/test/'

# %%
estimator = TensorFlow(entry_point = 'model.py',
                    source_dir = os.getcwd(),
                    role = role,
                    framework_version = '2.1.0',
                    train_instance_count = 2,
                    train_instance_type = 'ml.p2.xlarge',
                    py_version = 'py37',
                    output_path = f's3:/{bucket_name}/{prefix}/output'
                    )

# %%
estimator.fit({'train' : train_dir, 'test' : test_dir}, run_tensorboard_locally = True)


# %%
