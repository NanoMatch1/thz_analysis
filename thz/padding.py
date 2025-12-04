# -*- coding: utf-8 -*-
"""
Pad zeros around THz time-domain data so that the main peak
is centered in the time window, and then extend the trace
to a longer total duration.

Calls a windowing function on the centered trace.

@author: Marco Ballabio
"""

import pandas as pd
import numpy as np
import windowing as w
import matplotlib.pyplot as plt

def pad_zeros(df: pd.DataFrame, length_factor: int = 5) -> pd.DataFrame:
    """
    Symmetrically pads the dataset with zeros to extend its length by a specified factor. Perform after windowing.

    Parameters
    ----------
    df : DataFrame with ['Time (ps)', 'Mean', 'std error']
    length_factor : int
        Final length will be length_factor * original length.
    """

    time = df["Time (ps)"].to_numpy()
    y_mean = df["Mean"].to_numpy()
    std_err = df["std error"].to_numpy()
    N_array = len(time)

    if N_array < 3:
        return df

    dt = time[1] - time[0]

    pad_width = ((length_factor * N_array) - N_array) // 2
    end_value_left = time[0] - (pad_width * dt)
    end_value_right = time[-1] + (pad_width * dt)

    new_time = np.pad(time, (pad_width, pad_width), mode='linear_ramp', end_values=(end_value_left, end_value_right))
    new_y_mean =  np.pad(y_mean, (pad_width, pad_width), mode='constant', constant_values=(0,0))
    new_std_err = np.pad(std_err, (pad_width, pad_width), mode='constant', constant_values=(0,0))

    df_padded = pd.DataFrame({
        "Time (ps)": new_time,
        "Mean": new_y_mean,
        "std error": new_std_err,
    })

    return df_padded

def pad_to_window_range(df: pd.DataFrame, t_min: float, t_max: float) -> pd.DataFrame:
    """
    Pads the dataset with zeros to ensure the time axis spans from t_min to t_max.

    Parameters
    ----------
    df : DataFrame with ['Time (ps)', 'Mean', 'std error']
    t_min : float
        Desired minimum time value (ps).
    t_max : float
        Desired maximum time value (ps).
    """

    time = df["Time (ps)"].to_numpy()
    y_mean = df["Mean"].to_numpy()
    std_err = df["std error"].to_numpy()

    dt = time[1] - time[0]

    # Calculate required padding on each side
    pad_left = int(np.ceil((time[0] - t_min) / dt))
    pad_right = int(np.ceil((t_max - time[-1]) / dt))

    end_value_left = time[0] - (pad_left * dt)
    end_value_right = time[-1] + (pad_right * dt)

    new_time = np.pad(time, (pad_left, pad_right), mode='linear_ramp', end_values=(end_value_left, end_value_right))
    new_y_mean =  np.pad(y_mean, (pad_left, pad_right), mode='constant', constant_values=(0,0))
    new_std_err = np.pad(std_err, (pad_left, pad_right), mode='constant', constant_values=(0,0))

    df_padded = pd.DataFrame({
        "Time (ps)": new_time,
        "Mean": new_y_mean,
        "std error": new_std_err,
    })

    return df_padded

def edge_window_pad(df, alpha=0.2, padding_factor=5, padding=True):
    '''Uses edge-tapered windowing and no centering to create the windowed trace. Padding is done after windowing.'''
    # plt.plot(df['Time (ps)'], df['Mean'], label='original trace')
    df_windowed = w.edge_window(df, alpha=alpha)
    # padding with zeros
    if padding:
        df_windowed = pad_zeros(df_windowed, length_factor=padding_factor)
    return df_windowed

    # plt.plot(df['Time (ps)'], df['Mean'], label='windowed trace')
    # plt.legend()
    # plt.show()




def centerpad(df, length_factor=5):
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
    # desired_length = 5 * len(df)
    desired_length = length_factor * len(df)

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

    # plt.plot(df_padded['Time (ps)'], df_padded['Mean'], label='centered & padded trace')
    # plt.legend()
    # plt.show()
    return df_padded

def test_centerpad(time, y_mean):
    N_array = len(time)
    test_1 = y_mean.copy()
    test_2 = np.roll(y_mean, N_array//3)

    plt.plot(time, test_1, label='original trace'
             )
    plt.plot(time, test_2, label='rolled trace')
    plt.legend()
    plt.show()


def centerpad_refactor(df: pd.DataFrame, length_factor: int = 5) -> pd.DataFrame:
    """
    Center the main THz pulse in the time window, apply windowing,
    and extend the trace by zero-padding.

    Differences from older versions:
    - Does NOT crop data when centering: it preserves the full original trace.
    - Time axis is extended linearly using dt, no reflect tricks.

    Parameters
    ----------
    df : DataFrame with ['Time (ps)', 'Mean', 'std error']
    length_factor : int
        Final length will be approximately length_factor * original length.

    Returns
    -------
    df_out : DataFrame
    """

    df = df.copy()

    
    # 1) Remove DC offset from first 10 points
    if len(df) >= 10:
        df["Mean"] = df["Mean"] - df["Mean"].iloc[:10].mean()

    time = df["Time (ps)"].to_numpy()
    y_mean = df["Mean"].to_numpy()
    std_err = df["std error"].to_numpy()
    N_array = len(time)

    if N_array < 3:
        return df

    dt = time[1] - time[0]

    

    # 2) Find peak and desired center (original center index)
    peak_idx = int(np.argmax(np.abs(y_mean)))
    center_idx = N_array // 2
    difference = center_idx - peak_idx  # positive => pulse should move right, negative => left

    new_center = center_idx - difference
    new_length = new_center * 2

    # test_centerpad(time, y_mean)

    new_y_mean = np.roll(y_mean, shift*3)
    plt.plot(time, new_y_mean, label='rolled trace'
             )
    plt.show()

    # 3) Determine extra padding needed. Note the sign of the shift variable preserves direction, meaning we don't need an if/else here
    left_extra = max(shift, 0)
    right_extra = max(-shift, 0)

    end_value_left = time[0]-(left_extra * dt)
    end_value_right = time[-1]+(right_extra * dt)

    new_time = np.pad(time, (left_extra, right_extra), mode='linear_ramp', end_values=(end_value_left, end_value_right))
    new_y_mean = np.pad(y_mean, (left_extra, right_extra), mode='constant', constant_values=(0,0))
    new_std_err = np.pad(std_err, (left_extra, right_extra), mode='constant', constant_values=(0,0))

    plt.plot(time, y_mean, label="original trace", marker='o')
    plt.plot(new_time, new_y_mean, label='centered trace', marker='x')

    plt.legend()
    plt.show()

    breakpoint()

    # # New length after centering
    # N_centered = N_array + left_extra + right_extra

    # # 4) Build centered arrays
    # y_c = np.zeros(N_centered, dtype=float)
    # e_c = np.zeros(N_centered, dtype=float)

    # # new_y_mean = np.insert(y_mean, 0, np.zeros(left_extra))
    # # new_y_mean = np.append(new_y_mean, np.zeros(right_extra))

    # # breakpoint()

    # # put original data into the new array at the correct offset
    # start = left_extra
    # stop = left_extra + N_array
    # y_c[start:stop] = y_mean
    # e_c[start:stop] = std_err

    # # extend time linearly
    # t_left = t_c[0] - dt * np.arange(left_big, 0, -1)
    # t_right = t_c[-1] + dt * np.arange(1, right_big + 1)
    # t_final = np.concatenate([t_left, t_c, t_right])

    # plt.plot(time, y_mean, label='original trace', marker='o')
    # plt.plot(t_c, y_c, label='centered trace', marker='x')

    # # 5) Build new time axis linearly
    # # Put peak at the new center index
    # new_center_idx = N_centered // 2
    # t0_peak = time[peak_idx]
    # t_c = (np.arange(N_centered) - new_center_idx) * dt + t0_peak

    df_centered = pd.DataFrame({
        "Time (ps)": new_time,
        "Mean": new_y_mean,
        "std error": new_std_err,
    })

    # 6) Apply window on the centered trace
    # w.window(df_centered)

    # 7) Big symmetric zero-padding (length_factor)
    current_len = len(df_centered)
    target_len = int(current_len * length_factor)
    extra = max(target_len - current_len, 0)
    left_big = extra // 2
    right_big = extra - left_big

    # extend time linearly
    t_left = t_c[0] - dt * np.arange(left_big, 0, -1)
    t_right = t_c[-1] + dt * np.arange(1, right_big + 1)
    t_final = np.concatenate([t_left, t_c, t_right])

    # pad mean with zeros and stderr with edge values
    y_final = np.pad(df_centered["Mean"].to_numpy(),
                     (left_big, right_big),
                     mode="constant")

    e_final = np.pad(df_centered["std error"].to_numpy(),
                     (left_big, right_big),
                     mode="edge")
    
    plt.plot(t_final, y_final, label="centered & padded trace")
    plt.legend()
    plt.show()

    df_out = pd.DataFrame({
        "Time (ps)": t_final,
        "Mean": y_final,
        "std error": e_final,
    })

    return df_out
