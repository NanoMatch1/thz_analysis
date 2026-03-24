# thz/data_processing/preprocessing.py

import numpy as np
import pandas as pd
from scipy.signal.windows import tukey, hann   # change to hann if you prefer

from thz.data_structures.decorators import with_dataframe




# ---------- 1. Baseline subtraction ----------
def baseline_subtract(dataY: np.array, n_points: int = 10, **kwargs) -> np.array:
    """Subtract the mean of the first n_points from the y-values in the data array.
    
    Parameters
    ----------
    dataY : np.array
    1D array of y-values to baseline subtract.
    n_points : int
    Number of initial points to use for baseline calculation. Default is 10."""

    baseline = np.mean(dataY[:n_points])
    data_baselined = dataY - baseline
    return data_baselined

# ---------- 2. Windowing ----------
def edge_window(data: np.array, alpha=0.2, **kwargs) -> np.array:
    """
    Apply a Tukey window with edge tapering to the THz time-domain data.

    Parameters
    ----------
    data : np.array
        2D array with columns ['Time (ps)', 'Mean']
    alpha : float
        Shape parameter of the Tukey window (0 < alpha < 1).

    Returns
    -------
    data_windowed : np.array
        Copy of 2D array data with windowed 'Mean' and 'std error'.
    """
    dataX = data[:, 0]
    dataY = data[:, 1]
    std_err = data[:, 2]

    n_points = len(dataY)
    window = tukey(n_points, alpha)  # 5% edge taper
    windowed_data = dataY * window
    windowed_error = std_err * window

    data_windowed = np.column_stack((dataX, windowed_data, windowed_error))
    if kwargs.get('show_graph', False):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(data[:, 0], data[:, 1], label='Original', alpha=0.5, linewidth=3)
        plt.plot(data_windowed[:, 0], data_windowed[:, 1], label='Windowed')
        plt.xlabel('Time (ps)')
        plt.ylabel('Mean')
        plt.title('Edge Windowing with Tukey Window (alpha={})'.format(alpha))
        plt.legend()
        plt.grid()
        plt.show()

    return data_windowed


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

def pad_to_window_range(df: pd.DataFrame, time_window: tuple, pad_length_factor: float = 3, **kwargs) -> pd.DataFrame:
    """
    Pads the dataset with zeros to ensure the time axis spans from t_min to t_max.

    Parameters
    ----------
    df : DataFrame with ['Time (ps)', 'Mean', 'std error']
    time_window : tuple
        Desired time window as (t_min, t_max) in ps.
    pad_length_factor : int
        Additional padding factor to extend beyond the specified time window to ensure sufficient zero-padding for phase calculation.
    """

    if time_window is None:
        raise ValueError("time_window must be provided as (t_min, t_max)")
    
    # kwargs['show_graph'] = True

    time = df["Time (ps)"].to_numpy()
    y_mean = df["Mean"].to_numpy()
    std_err = df["std error"].to_numpy()

    dt = time[1] - time[0]

    t_min, t_max = time_window
    full_window = t_max - t_min
    extended_window = full_window * pad_length_factor
    t_min = (t_min + t_max)/2 - extended_window/2
    t_max = (t_min + t_max)/2 + extended_window/2

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

    if kwargs.get('show_graph', False):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(df['Time (ps)'], df['Mean'], label='Original', alpha=0.5, linewidth=4)
        plt.plot(df_padded['Time (ps)'], df_padded['Mean'], label='Padded')
        plt.xlabel('Time (ps)')
        plt.ylabel('Mean')
        plt.title('Zero Padding to Specified Time Window')
        plt.legend()
        plt.grid()
        plt.show()

    return df_padded



# ---------- 4. Full preprocessing pipeline ----------
@with_dataframe(columns=["Time (ps)", "Mean", "std error"])
def preprocess_trace(
    df: pd.DataFrame,
    baseline_points: int = 10,
    window_alpha: float = 0.2,
    pad_length_factor: int = 2,
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
    # df_processed = pad_zeros(df_windowed, length_factor=pad_length_factor)
    

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
