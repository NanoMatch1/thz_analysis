# -*- coding: utf-8 -*-
"""
Importing 2-column data, tab separated, with an arbitrary number of % commented lines
Returns a dataframe with x as first column and a y column for each measured scan 

Created on Wed June 06 2023

@author: Marco Ballabio
"""
import numpy as np
import pandas as pd
from os import sep

def dataimport(filepath,filename):

    '''Imports and processes the data file into a pandas DataFrame, with columns for time, mean, and standard error.'''
   
    # # Read the file and split it into lines
    with open(filepath+sep+filename, 'r') as file:
        lines = file.readlines()
    
    # Initialize empty lists for x and y values
    x_values = []
    y_values = []
    
    # Process the lines
    for line in lines:
        # Skip commented lines
        if line.startswith('%'):
            continue
    
        # Split the line into values
        values = line.split()
    
        # Check the number of values
        if len(values) == 2:
            # Extract x and y values
            x = float(values[0])
            y = float(values[1])
    
            # Append x and y values to the lists
            x_values.append(x)
            y_values.append(y)
    
    # Convert lists to numpy arrays
    x_array = np.array(np.unique(x_values))
    # x_array = x_array-min(x_array)
    y_array = np.array(y_values)
    
    # Determine the number of datasets
    num_datasets = len(x_array)
    
    # Reshape y_array to match the number of datasets
    y_array = y_array.reshape(-1, num_datasets)
    y_array=np.swapaxes(y_array, 0, 1)
    
    # Create a DataFrame from the arrays
    df = pd.DataFrame(np.column_stack((x_array, y_array)))
    df = df.rename(columns={0: 'Time (ps)'})
    df.set_index('Time (ps)')
    
    #calculate mean and standard error
    y_values = df.iloc[:,1:]
    y_mean = y_values.mean(1)
    y_err = y_values.sem(1)
    
    #cancel offset, computed on first 10 points of time trace
    y_mean = y_mean-np.average(y_mean[0:10])
    
    #add in dataframe after time column    
    df.insert(1, 'Mean', y_mean, allow_duplicates = False)
    df.insert(2, 'std error', y_err, allow_duplicates = False)
    df.drop(df.columns[3:], axis=1, inplace = True) #drop single scans
    
    return df
