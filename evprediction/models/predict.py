# %%
import numpy as np
import pandas as pd 

# %%
def convert_to_array(arr):
    """This takes in the raw data and generates arrays specific to LSTMs
    
    Arguments:
        arry {np.array} -- A row of 2880 data points. (1/2 hour interval for 2 months)
    
    Returns:
        X_test_arr {np.array} -- reshaped testing data
    """    

    # Convert to np array and reshape into LSTM format
    X_test_arr = np.reshape(arr, (2880, 60, -1))

    return X_test_arr