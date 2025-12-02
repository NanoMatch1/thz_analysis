# thz/data_processing/preprocessing.py

import numpy as np
import pandas as pd
from scipy.signal.windows import tukey   # change to hann if you prefer

# If you want to support ndarray via your decorator:
from thz.data_structures.decorators import with_dataframe
# and decorate the public function at the bottom


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
    df_out = df.copy()
    if len(df_out) < n_points:
        return df_out

    offset = df_out['Mean'].iloc[:n_points].mean()
    df_out['Mean'] = df_out['Mean'] - offset
    return df_out


# ---------- 2. Center main pulse in time window ----------

def center_pulse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Center the main THz pulse (max |Mean|) in the time window by shifting
    the data in index space and padding the vacated region with zeros.

    The length and time step are preserved.

    Returns
    -------
    df_centered : DataFrame
    """
    df = df.copy()

    t = df['Time (ps)'].to_numpy()
    y = df['Mean'].to_numpy()
    e = df['std error'].to_numpy()

    N = len(t)
    if N < 3:
        return df

    # assume roughly uniform dt
    dt = t[1] - t[0]

    peak_idx = int(np.argmax(np.abs(y)))
    center_idx = N // 2
    shift = center_idx - peak_idx  # +ve => move pulse to the right

    y_c = np.zeros_like(y)
    e_c = np.zeros_like(e)

    if shift > 0:
        # pulse moves right; left side becomes zeros
        y_c[shift:] = y[:-shift]
        e_c[shift:] = e[:-shift]
    elif shift < 0:
        # pulse moves left; right side becomes zeros
        y_c[:shift] = y[-shift:]
        e_c[:shift] = e[-shift:]
    else:
        y_c[:] = y
        e_c[:] = e

    # build centered time axis, symmetric around the pulse time
    t0 = t[peak_idx]
    t_c = (np.arange(N) - center_idx) * dt + t0

    df_centered = pd.DataFrame(
        {
            "Time (ps)": t_c,
            "Mean": y_c,
            "std error": e_c,
        }
    )
    return df_centered


# ---------- 3. Gentle window ----------

def apply_window(
    df: pd.DataFrame,
    alpha: float = 0.2,
) -> pd.DataFrame:
    """
    Apply a gentle Tukey window to the trace.

    Parameters
    ----------
    df : DataFrame
    alpha : float
        Tukey window shape parameter (0 <= alpha <= 1).
        - alpha ~ 0.0 -> almost boxcar (flat)
        - alpha ~ 0.2 -> 20% cosine tapers at edges
        - alpha ~ 1.0 -> Hann

    Returns
    -------
    df_windowed : DataFrame
    """
    df_w = df.copy()
    N = len(df_w)
    if N < 4:
        return df_w

    w = tukey(N, alpha=alpha)

    y = df_w['Mean'].to_numpy()
    e = df_w['std error'].to_numpy()

    df_w['Mean'] = y * w
    df_w['std error'] = e * w

    return df_w


# ---------- 4. Zero-padding to longer length ----------

def pad_zeros(
    df: pd.DataFrame,
    length_factor: float = 10.0,
) -> pd.DataFrame:
    """
    Zero-pad the trace on both sides to increase total length.

    The time step is preserved and the time axis is extended linearly.

    Parameters
    ----------
    df : DataFrame
    length_factor : float
        Target length ≈ length_factor * original_length.
        If <= 1, no padding is applied.

    Returns
    -------
    df_padded : DataFrame
    """
    df = df.copy()
    t = df['Time (ps)'].to_numpy()
    y = df['Mean'].to_numpy()
    e = df['std error'].to_numpy()

    N = len(t)
    if N < 3 or length_factor <= 1.0:
        return df

    dt = t[1] - t[0]
    target_len = int(np.ceil(N * length_factor))
    extra = max(target_len - N, 0)
    if extra == 0:
        return df

    left_extra = extra // 2
    right_extra = extra - left_extra

    # extend time axis
    t_left = t[0] - dt * np.arange(left_extra, 0, -1)
    t_right = t[-1] + dt * np.arange(1, right_extra + 1)
    t_ext = np.concatenate([t_left, t, t_right])

    # pad data
    y_ext = np.pad(y, (left_extra, right_extra), mode='constant')
    # for stderr we can either pad with zeros or repeat edge values;
    # here we repeat edges so error doesn't artificially go to 0
    e_ext = np.pad(e, (left_extra, right_extra), mode='edge')

    df_padded = pd.DataFrame(
        {
            "Time (ps)": t_ext,
            "Mean": y_ext,
            "std error": e_ext,
        }
    )
    return df_padded


# ---------- 5. One-shot convenience function ----------

# If you want ndarray support, uncomment the decorator and import it
@with_dataframe(columns=["Time (ps)", "Mean", "std error"])
def centerpad_window(
    df: pd.DataFrame,
    length_factor: float = 10.0,
    baseline_points: int = 10,
    window_alpha: float = 0.2,
) -> pd.DataFrame:
    """
    Full preprocessing for THz time-domain trace:

        1) subtract baseline from early-time points
        2) center main pulse in time window
        3) apply gentle window
        4) zero-pad to longer length

    Parameters
    ----------
    df : DataFrame (or ndarray if using @with_dataframe)
    length_factor : float
        Final length ≈ length_factor * original_length.
    baseline_points : int
        Number of points at start used for baseline estimate.
    window_alpha : float
        Tukey window alpha parameter (see apply_window).

    Returns
    -------
    df_out : DataFrame
    """
    df0 = baseline_subtract(df, n_points=baseline_points)
    df1 = center_pulse(df0)
    # df2 = apply_window(df1, alpha=window_alpha)
    # df3 = pad_zeros(df1, length_factor=length_factor)

    import matplotlib.pyplot as plt

    plt.plot(df['Time (ps)'], df['Mean'], label='original')
    plt.plot(df1['Time (ps)'], df1['Mean'], label='processed')
    plt.legend()
    plt.show()
    return df1