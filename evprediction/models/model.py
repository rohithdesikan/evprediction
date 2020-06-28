## THIS SCRIPT IS USED TO PREDICT WHETHER OR NOT A HOUSE HAS AN ELECTRIC VEHICLE
# %%
# Import base packages
import os
import numpy as np
import pandas as pd
import time
import yaml
import datetime
from IPython.display import display

# Pandas Formatting and Styling:
pd.options.display.max_rows = 200
pd.options.display.max_columns = 500
pd.set_option('display.float_format',lambda x: '%.3f' % x)

# Data Visualization Packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'image.cmap': 'cubehelix'})
sns.set_context('poster')

# ML Packages
# Import Machine Learning Packages (SKLearn and Tensorflow)
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, ensemble, metrics
from sklearn.preprocessing import normalize, MinMaxScaler

# Import Tensorflow
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam

# Set up the config file with hyperparameters
config_file = 'config.yml'

# Read in YAML file
with open(config_file, 'r') as f:
    inputs = yaml.load(f,Loader=yaml.FullLoader)

# %%
datadir = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', 'external'))


# %%
# FUNCTION 1: READ IN FILES
def read_files(inputs):
    """Reads inputs from a specified config file set up as shown in the config.yml file in the same folder
    
    Arguments:
        inputs {dict} -- This has specified paths to the relevant datasets
    
    Returns:
        [pd.DataFrame] -- This is the full dataset of meter readings read in from the current working directory and the path specific in the config file
        [pd.DataFrame] -- This is the full dataset of labels read in from the current working directory and the path specific in the config file
    """    

    # Read in Dataset else make sure the data paths are correct
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    try:
        X_full_path = os.path.join(curr_dir, inputs['data']['X_arr'])
        y_full_path = os.path.join(curr_dir, inputs['data']['y_arr'])
    except:
        raise NameError("Check the config file to make sure the data paths are placed under the data heading")

    ev_train = pd.read_csv(X_full_path)
    ev_train_labels = pd.read_csv(y_full_path)

    return ev_train, ev_train_labels


# FUNCTION 2: GENERATE LSTM ARRAYS WITH TIME STEPS AND THE 4 RELEVANT ARRAYS
def generate_arrays(time_steps, X, y_hot, testsize):
    """This takes in the raw data and generates arrays specific to LSTMs
    
    Arguments:
        time_steps {int} -- HYPERPARAMETER: This is specific in the config file and can be changed there. What number of time steps should the LSTM look back on?
        X {pd.DataFrame} -- This is the FULL dataset. A validation set is created within this function
        y_hot {pd.DataFrame} -- This is the FULL set of labels. A validation set is created within this function. 
        testsize {float} -- What should the size of the test set be? This is also specified in the config file. 
    
    Returns:
        X_arr_train {np.array} -- Training Data 
        y_arr_train {np.array} -- Training Labels
        X_arr_val {np.array} -- Validation Data 
        y_arr_val {np.array} -- Validation Labels
    """    

    # Train test split according to testsize from config file
    Xt, Xv, yt, yv = train_test_split(X, y_hot, test_size=testsize, random_state=42)

    # Convert to np array and reshape into LSTM format
    Xarr_train = np.array(Xt).reshape(len(Xt), time_steps, -1)
    Xarr_val = np.array(Xv).reshape(len(Xv), time_steps, -1)

    # Set the y labels as np arrays
    yarr_train = np.array(yt.copy())
    yarr_val = np.array(yv.copy())

    return Xarr_train, yarr_train, Xarr_val, yarr_val

# FUNCTION 3: CREATE THE ACTUAL LSTM NETWORK WITH THE TIME STEPS AND ACTIVATIONS ETC
def create_lstm(num_lstm_cells,time_steps):
    """Creates the multi layer LSTM model. The number of dense units before and after the LSTM layers can be changed here. 
    
    Arguments:
        num_lstm_cells {int} -- HYPERPARAMETER: This is specified in the config file and can be changed there. How many LSTM cells are needed?
        time_steps {int} -- HYPERPARAMETER: This is specific in the config file and can be changed there. What number of time steps should the LSTM look back on?
    
    Returns:
        [keras model] -- This returns a keras sequential model. 
    """

    # Create the sequential LSTM model
    model = Sequential()
    model.add(Dense(units = 100, activation = 'relu'))
    model.add(LSTM(num_lstm_cells,
                                input_shape = (time_steps, 1),
                                return_sequences = True))
    model.add(LSTM(num_lstm_cells, 
                                input_shape = (time_steps, 1),
                                return_sequences = False))
    model.add(Dense(units = 100, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))

    return model

## FUNCTION 4: MODEL TRAINING FUNCTION
def train_model(X_train, y_train, X_val, y_val, model, training_epochs, learning_rate):
    """Train the model using binary cross entropy loss (whether or not the house has an EV) using the Adam optimizer with a constant learning rate
    
    Arguments:
        model {keras model} -- This the model created in the last function
        training_epochs {int} -- HYPERPARAMETER: This is specified in the config file. 
    
    Returns:
        [keras model history] -- This is the results of the keras model
    """    
    model.compile(loss = keras.losses.binary_crossentropy, 
              optimizer = Adam(lr=learning_rate, 
                                beta_1=0.9, 
                                beta_2=0.999, 
                                epsilon=None, 
                                decay=0.002, 
                                amsgrad=False), 
              metrics = ['accuracy'])

    history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = training_epochs)

    return history

########################################################################################################

# %%
# if __name__ == "__main__":
# Read in files, run the clean function and generate single y labels from the timed y data
## Create new dataframes to work with
ev_train, ev_train_labels = read_files(inputs) 
X = ev_train.copy()
y = ev_train_labels.copy()

# Use the ETL script to clean the data
X, y = etl.clean_data(X, y)
y_hot = etl.generate_labels(y)

# This set of try except statements catches errors in the yml file. Certain parameters must be specified. 
try:
    # Run the generate LSTM arrays function
    num_lstm_cells=inputs['model1_hps']['num_lstm_cells']

    # How many time steps should be seen back?
    time_steps=inputs['model1_hps']['time_steps']

    # What is the test size of validation?
    valset_size = inputs['model1_hps']['test_size']

    # How many epochs of training?
    training_epochs = inputs['model1_hps']['epochs'] 

    # What is the learning rate?
    learning_rate = inputs['model1_hps']['learning_rate']

    # Where should the model be saved? It will be saved in that folder as specified in the config file. Generally, 'experiments'
    model_path = inputs['models']['model_path']

except:
    raise NameError("Check under the model1_hps heading to make sure it contains num_lstm_cells, time_steps, test_size, epochs and learning rate")

Xarr_train, yarr_train, Xarr_val, yarr_val = generate_arrays(time_steps, 
                                                            X, 
                                                            y_hot, 
                                                            testsize=valset_size)

# Check shapes if needed
# display(Xarr_train.shape)
# display(Xarr_val.shape)
# display(yarr_train.shape)
# display(yarr_val.shape)

# Simply create the LSTM, train teh model and obtain accuracies and model summary
model = create_lstm(num_lstm_cells,time_steps)
history = train_model(Xarr_train, yarr_train, Xarr_val, yarr_val, model, training_epochs, learning_rate)
print(model.summary())

# %%
# Save the model in the specified folder
curr_dir = os.path.dirname(os.path.realpath(__file__))
model_fn = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')}_trained_model.h5"
model_dir = os.path.join(curr_dir, model_path, model_fn)
model.save(model_dir)

# %%
# Retrieving the traiend model from Azure
azure_subscription_id = os.environ['Azure_Subscription_id']
ws = Workspace(azure_subscription_id, resource_group='data',workspace_name='EV_prediction')
experiment = Experiment(ws,'EV-predict-q1-exp3')
# print(experiment)


# %%

# FOR PREDICTIONS (TO TRANSFER TO THE PREDICT.PY FILE)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])

# # %%
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])

# %%
