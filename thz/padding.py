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
import windowing as w


def centerpad(df: pd.DataFrame, length_factor: int = 10) -> pd.DataFrame:
    """
    Center the main THz peak in time by padding with zeros and
    extend the trace length by a given multiplicative factor.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'Time (ps)', 'Mean', 'std error'
    length_factor : int, optional
        Final length will be approximately `length_factor * N`,
        where N is the original number of points. Default is 10.

    Returns
    -------
    df_padded : pd.DataFrame
        DataFrame with zero-padded 'Mean' and 'std error', and
        time padded by odd reflection.
    """
    # Work on a copy so we don't modify the original
    df = df.copy()

    # 1. Remove DC offset from first 10 points
    df['Mean'] = df['Mean'] - df['Mean'].iloc[:10].mean()

    # 2. Extract arrays
    mean = df['Mean'].to_numpy()
    err = df['std error'].to_numpy()
    time = df['Time (ps)'].to_numpy()

    # 3. Find peak index and how much we need to shift it to center
    peak_index = np.argmax(np.abs(mean))
    center_index = len(mean) // 2
    shift = center_index - peak_index  # +ve => pad on the left, -ve => pad on the right

    if shift > 0:
        # Need to pad at the start (left)
        pad_width = (shift, 0)
        slice_obj = slice(0, len(mean))  # keep first N entries after padding
    elif shift < 0:
        # Need to pad at the end (right)
        pad_width = (0, -shift)
        slice_obj = slice(-shift, None)  # drop first |shift| entries
    else:
        # Already centered
        pad_width = (0, 0)
        slice_obj = slice(None)

    # 4. Apply padding and slicing consistently
    mean = np.pad(mean, pad_width, mode='constant')[slice_obj]
    err = np.pad(err, pad_width, mode='constant')[slice_obj]
    time = np.pad(time, pad_width, mode='reflect', reflect_type='odd')[slice_obj]

    # 5. Build centered DataFrame and apply window
    df_centered = pd.DataFrame({
        'Time (ps)': time,
        'Mean': mean,
        'std error': err
    })

    # Assuming w.window operates in-place on df_centered
    w.window(df_centered)

    # 6. Extend length by padding zeros on both sides
    current_len = len(df_centered)
    desired_length = length_factor * current_len
    extra_total = max(desired_length - current_len, 0)

    left_extra = extra_total // 2
    right_extra = extra_total - left_extra  # handle odd differences

    mean_ext = np.pad(df_centered['Mean'].to_numpy(),
                      (left_extra, right_extra),
                      mode='constant')
    err_ext = np.pad(df_centered['std error'].to_numpy(),
                     (left_extra, right_extra),
                     mode='constant')
    time_ext = np.pad(df_centered['Time (ps)'].to_numpy(),
                      (left_extra, right_extra),
                      mode='reflect',
                      reflect_type='odd')

    df_padded = pd.DataFrame({
        'Time (ps)': time_ext,
        'Mean': mean_ext,
        'std error': err_ext
    })

    return df_padded
