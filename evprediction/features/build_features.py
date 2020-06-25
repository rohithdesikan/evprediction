import numpy as np
import pandas as pd

def clean_data(X,y):
    """Takes data as a csv file, gets rid of the house id column and removes rows with NaN values
    
    Arguments:
        X {pd.DataFrame} -- This is the full dataset of meter readings at each time step. This will be split up into training and validation later
        y {pd.DataFrame} -- These are the labels at each time step
    
    Returns:
        [pd.DataFrame] -- This is the new cleaned X matrix
        [pd.DataFrame] -- This is the cleaned y matrix
    """

    # Drop the House ID from both x and y
    X.drop('House ID', axis = 1, inplace = True)
    y.drop('House ID', axis = 1, inplace = True)

    #Find the rows that have NaNs, there are only 4, so we can get rid of them. 
    nan_ind = X.isna().sum(axis = 1).nonzero()
    # X.iloc[nan_ind[0],:]
    X_cleaned = X.drop(nan_ind[0], axis = 0, inplace = False)
    y_cleaned = y.drop(nan_ind[0], axis = 0, inplace = False)

    return X_cleaned, y_cleaned


def generate_labels(y):
    """Generates labels for the 1st task as to whether or not a single house has an EV
    
    Arguments:
        y {pd.DataFrame} -- This is the cleaned labels
    
    Returns:
        [np.array] -- A single column of y vectors where 0 means no EV and 1 means the house has an EV
    """    
    y_hot = np.array([1 if a > 0 else 0 for a in y.sum(axis = 1).values.tolist()]).reshape(-1,1)

    return y_hot