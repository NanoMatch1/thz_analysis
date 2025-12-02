# -*- coding: utf-8 -*-
"""
Pad zeros around THz time-domain data so that the main peak
is centered in the time window, and then extend the trace
to a longer total duration.

Calls a windowing function on the centered trace.

@author: Marco Ballabio
Refactored by: ChatGPT/Samuel Brooke 11/24/25
"""

import pandas as pd
import numpy as np
import thz.data_processing.preprocessing as w
from thz.data_structures.decorators import with_dataframe


def centerpad(df, length_factor=10):
    #remove offset from Mean column
    df['Mean'] = df['Mean']-np.average(df['Mean'].values[0:10])    
    
    # Find the index of the peak in the amplitude column (using absolute values)
    peak_index = np.argmax(np.abs(df['Mean']))
    
    # Calculate the number of zeros to add on either side
    num_zeros_to_add = (len(df) // 2 - peak_index)
    
    # if num_zeros_to_add == 0:
    #     return df
    
   # time_step = df['Time (ps)'].iloc[1] - df['Time (ps)'].iloc[0]
    if num_zeros_to_add < 0:
        num_zeros_to_add = abs(num_zeros_to_add)
        # Add zeros at the end of both columns while removing the first values
        # time flows forwards
        padded_mean = np.pad(df['Mean'].values, (0, num_zeros_to_add), mode='constant')
        padded_error = np.pad(df['std error'].values, (0, num_zeros_to_add), mode='constant')
        padded_time = np.pad(df['Time (ps)'].values, (0, num_zeros_to_add), mode='reflect',reflect_type='odd')
        padded_mean = padded_mean[num_zeros_to_add:]
        padded_error = padded_error[num_zeros_to_add:]
        padded_time = padded_time[num_zeros_to_add:]
    elif num_zeros_to_add > 0:
        # Add zeros at the beginning of both columns while removing the last values
        # time flows backwards
        num_zeros_to_add = abs(num_zeros_to_add)
        padded_mean = np.pad(df['Mean'].values, (num_zeros_to_add, 0), mode='constant')
        padded_error = np.pad(df['std error'].values, (num_zeros_to_add, 0), mode='constant')
        padded_time = np.pad(df['Time (ps)'].values, (num_zeros_to_add, 0), mode='reflect',reflect_type='odd')
        # padded_time = np.pad(df['Time (ps)'].values, (num_zeros_to_add, 0), mode='reflect',reflect_type='odd')
        padded_mean = padded_mean[:-num_zeros_to_add]
        padded_time = padded_time[:-num_zeros_to_add]
        padded_error = padded_error[:-num_zeros_to_add]
        
    else:
        padded_mean = df['Mean'].values
        padded_error = df['std error'].values
        padded_time =df['Time (ps)'].values
            
    # Create a new DataFrame with the zero-padded amplitude
    df_padded = pd.DataFrame({'Time (ps)' : padded_time,
                            'Mean' : padded_mean,
                            'std error': padded_error})
    
    w.window(df_padded)
    
    #increase size by padding zeroes, it helps when pulses arrival time difference is large (thick samples)
    desired_length = length_factor * len(df_padded)

    # Calculate the number of zeros to add on each side
    zeros_to_add = (desired_length - len(df)) // 2

    # Pad the array with zeros at both sides
    padded0_mean = np.pad(df_padded['Mean'], (zeros_to_add, zeros_to_add), mode='constant')
    padded0_error = np.pad(df_padded['std error'], (zeros_to_add, zeros_to_add), mode='constant')
    padded0_time = np.pad(padded_time, (zeros_to_add, zeros_to_add), mode='reflect',reflect_type='odd')

        
    # Create a new DataFrame with the zero-padded amplitude
    df_padded = pd.DataFrame({'Time (ps)' : padded0_time,
                            'Mean' : padded0_mean,
                            'std error': padded0_error})
    # w.window(df_padded)
    return df_padded
