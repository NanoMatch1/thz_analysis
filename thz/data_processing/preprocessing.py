# thz/data_processing/preprocessing.py

import numpy as np
import pandas as pd
from scipy.signal.windows import tukey   # change to hann if you prefer

from thz.data_structures.decorators import with_dataframe



# ---------- 1. Baseline subtraction ----------

def baseline_subtract(
    df: pd.DataFrame,
    n_points: int = 10,
) -> pd.DataFrame:
    """
    Subtract a DC offset estimated from the first `n_points` of the trace.

    Parameters
    ----------
    df : DataFrame with ['Time (ps)', 'Mean', 'std error']
    n_points : int
        Number of initial points to use for the baseline estimate.

    Returns
    -------
    df_out : DataFrame
        Copy of df with 'Mean' baseline-corrected.
    """
    df_baseline = df.copy()
    if len(df_baseline) < n_points:
        return df_baseline

    offset = df_baseline['Mean'].iloc[:n_points].mean()
    df_baseline['Mean'] = df_baseline['Mean'] - offset
    return df_baseline


# ---------- 2. Windowing ----------
@with_dataframe(columns=["Time (ps)", "Mean", "std error"])
def edge_window(df, alpha=0.2) -> pd.DataFrame:
    """
    Apply a Tukey window with edge tapering to the THz time-domain data.

    Parameters
    ----------
    df : DataFrame with ['Time (ps)', 'Mean', 'std error']
    alpha : float
        Shape parameter of the Tukey window (0 < alpha < 1).

    Returns
    -------
    df_windowed : DataFrame
        Copy of df with windowed 'Mean' and 'std error'.
    """
    df_windowed = df.copy()
    y_mean = df_windowed['Mean'].values
    std_err = df_windowed['std error'].values
    N = len(y_mean)
    w = tukey(N, alpha)  # 5% edge taper
    windowed_data = y_mean * w
    windowed_error = std_err * w

    df_windowed['Mean'] = windowed_data
    df_windowed['std error'] = windowed_error
    return df_windowed


# ---------- 3. Padding ----------
@with_dataframe(columns=["Time (ps)", "Mean", "std error"])
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


# ---------- 4. Full preprocessing pipeline ----------
@with_dataframe(columns=["Time (ps)", "Mean", "std error"])
def preprocess_trace(
    df: pd.DataFrame,
    baseline_points: int = 10,
    window_alpha: float = 0.2,
    pad_length_factor: int = 5,
    show_graph: bool = False,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline: baseline subtraction, windowing, and zero-padding.

    Parameters
    ----------
    df : DataFrame with ['Time (ps)', 'Mean', 'std error']
    baseline_points : int
        Number of initial points to use for baseline subtraction.
    window_alpha : float
        Shape parameter of the Tukey window (0 < alpha < 1).
    pad_length_factor : int
        Final length will be pad_length_factor * original length.

    Returns
    -------
    df_processed : DataFrame
        Preprocessed DataFrame.
    """
    df_baselined = baseline_subtract(df, n_points=baseline_points)
    df_windowed = edge_window(df_baselined, alpha=window_alpha)
    df_processed = pad_zeros(df_windowed, length_factor=pad_length_factor)

    if show_graph:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(df['Time (ps)'], df['Mean'], label='Original', alpha=0.5)
        plt.plot(df_baselined['Time (ps)'], df_baselined['Mean'], label='Baselined', alpha=0.8)
        plt.plot(df_windowed['Time (ps)'], df_windowed['Mean'], label='Windowed', alpha=0.8)
        plt.plot(df_processed['Time (ps)'], df_processed['Mean'], label='Padded', alpha=0.8)
        plt.xlabel('Time (ps)')
        plt.ylabel('Mean')
        plt.title('THz Trace Preprocessing')
        plt.legend()
        plt.grid()
        plt.show()

    return df_processed
