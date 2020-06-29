## THIS SCRIPT IS USED TO PREDICT WHETHER OR NOT A HOUSE HAS AN ELECTRIC VEHICLE
# %%
# Import base packages
import os
import numpy as np
import pandas as pd
import argparse

# ML Packages
# Import Machine Learning Packages
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout
from keras.layers import LSTM
from keras.optimizers import Adam


# %%
def generate_train_array(train_dir, time_steps):
    """This takes in the raw data and generates arrays specific to LSTMs
    
    Arguments:
        train_dir {str} -- The training directory, evironment variable in Sagemaker
        time_steps {int} -- The number of timesteps in the LSTM model (HYPERPARAMETER)
    
    Returns:
        X_train_arr {np.array} -- Training Data 
        y_train_arr {np.array} -- Training Labels
    """    

    print("Training Directory: ", train_dir)
    X_train_path = os.path.join(train_dir, 'X_train.csv')
    y_train_path = os.path.join(train_dir, 'y_train.csv')

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)

    # Convert to np array and reshape into LSTM format
    X_train_arr = np.reshape(np.array(X_train), (len(X_train), time_steps, -1))

    # Set the y labels as np arrays
    y_train_arr = np.array(y_train)

    return X_train_arr, y_train_arr

def generate_test_array(test_dir, time_steps):
    """This takes in the raw data and generates arrays specific to LSTMs
    
    Arguments:
        test_dir {str} -- The testing directory, evironment variable in Sagemaker
        time_steps {int} -- The number of timesteps in the LSTM model (HYPERPARAMETER)
    
    Returns:
        X_test_arr {np.array} -- testing Data 
        y_test_arr {np.array} -- testing Labels
    """    

    print("Test Directory: ", test_dir)
    X_test_path = os.path.join(test_dir, 'X_test.csv')
    y_test_path = os.path.join(test_dir, 'y_test.csv')

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    # Convert to np array and reshape into LSTM format
    X_test_arr = np.reshape(np.array(X_test), (len(X_test), time_steps, -1))

    # Set the y labels as np arrays
    y_test_arr = np.array(y_test)

    return X_test_arr, y_test_arr


def create_lstm(num_dense_units, num_lstm_cells, time_steps):
    """Creates the multi layer LSTM model. The number of dense units before and after the LSTM layers can be changed here. 
    
    Arguments:
        num_lstm_cells {int} -- HYPERPARAMETER: This is specified in the config file and can be changed there. How many LSTM cells are needed?
        time_steps {int} -- HYPERPARAMETER: This is specific in the config file and can be changed there. What number of time steps should the LSTM look back on?
    
    Returns:
        [keras model] -- This returns a keras sequential model. 
    """

    # Create the sequential LSTM model
    model = Sequential()
    model.add(Dense(units = num_dense_units, activation = 'relu'))
    model.add(LSTM(num_lstm_cells,
                                input_shape = (time_steps, 1),
                                return_sequences = True))
    model.add(LSTM(num_lstm_cells, 
                                input_shape = (time_steps, 1),
                                return_sequences = False))
    model.add(Dense(units = num_dense_units, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))

    return model

def train(X_train, y_train, X_test, y_test, model, epochs, learning_rate):
    """Train the model using binary cross entropy loss (whether or not the house has an EV) using the Adam optimizer with a constant learning rate
    
    Arguments:
        X_train {int} -- X train array
        y_train {int} -- y train array
        X_test {int} -- X test array
        y_test {int} -- y test array
        model {keras model} -- Keras LSTM Model created above
        epochs {int} -- HYPERPARAMETER: spochs of training
        epochs {int} -- HYPERPARAMETER: learning rate for Adam optimizer
    
    Returns:
        [history] {keras model history} -- This is the results of the keras model
    """    
    model.compile(loss = keras.losses.binary_crossentropy, 
              optimizer = Adam(lr=learning_rate), 
              metrics = ['accuracy'])

    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epochs)

    return history

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add the necessary arguments
    parser.add_argument('--num-dense-units', type=int, default=100, metavar='N',
                        help='Number of dense units for dense layers (default: 100)')

    parser.add_argument('--num-lstm-cells', type=int, default=128, metavar='N',
                        help='Number of LSTM hidden units for recurrent layers (default: 128)')

    parser.add_argument('--time-steps', type=int, default=60, metavar='N',
                        help='The number of time steps that this LSTM model runs over.')

    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 40)')

    parser.add_argument('--learning-rate', type=float, default=0.01, metavar='S',
                        help='learning rate (default: 0.01)')
    
    # Data Directories
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test-dir', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args = parser.parse_args()

    # Run the functions to train the model
    X_train, y_train = generate_train_array(args.train_dir, args.time_steps)
    X_test, y_test = generate_test_array(args.test_dir, args.time_steps)
    model = create_lstm(args.num_dense_units, args.num_lstm_cells, args.time_steps)
    history = train(X_train, y_train, X_test, y_test, model, args.epochs, args.learning_rate)

    model.save(os.environ['SM_MODEL_DIR'] + r'/evmodel.h5')
    

# %%
