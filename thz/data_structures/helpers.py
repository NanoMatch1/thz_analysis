import pandas as pd
import numpy as np

def df_to_array(df: pd.DataFrame) -> np.ndarray:
    """Convert a pandas DataFrame to a numpy ndarray - columns not preserved."""
    return df.to_numpy()

def df_to_dict(df: pd.DataFrame) -> dict:
    """Convert a pandas DataFrame to a the dictionary format with 'data' and 'headers' keys."""
    # return {col: df[col].to_numpy() for col in df.columns}
    return {
        'data': df.to_numpy(),
        'headers': df.columns.tolist()
    }

def is_monotonic(x):
    d = np.diff(x)
    return np.all(d >= 0) or np.all(d <= 0)


def dict_to_df(data_dict: dict) -> pd.DataFrame:
    """Convert a dictionary of numpy ndarrays to a pandas DataFrame."""
    data = data_dict['data']
    headers = data_dict['headers']
    return pd.DataFrame(data, columns=headers)

def array_to_df(array: np.ndarray, columns: list[str] | None = None) -> pd.DataFrame:
    """Convert a numpy ndarray to a pandas DataFrame."""
    return pd.DataFrame(array, columns=columns)



def _extract_data_and_headers(obj):
    """
    Convert supported objects into a canonical (data, headers, fmt) tuple.

    Supported:
      - pandas.DataFrame
      - dict with keys {'data': ndarray, 'headers': list}

    Unsupported:
      - bare numpy.ndarray (raises TypeError)
      - anything else
    """
    if isinstance(obj, pd.DataFrame):
        data = obj.to_numpy()
        headers = obj.columns.tolist()
        fmt = "dataframe"

    elif isinstance(obj, dict) and "data" in obj and "headers" in obj:
        data = obj["data"]
        headers = obj["headers"]
        fmt = "dict"

        if not isinstance(data, np.ndarray):
            raise TypeError("dict['data'] must be a numpy array")
        if not isinstance(headers, (list, tuple)):
            raise TypeError("dict['headers'] must be a list of column names")
        if data.ndim != 2:
            raise ValueError("dict['data'] must be a 2D array")

    else:
        raise TypeError(
            f"Unsupported type {type(obj)} — only DataFrame or "
            "dict{'data': ndarray, 'headers': list} are allowed."
        )

    return data, headers, fmt


def interpolate_to_max_resolution(
    *args,
    axis_col_name: str = None,      # must be provided for header safety
    clip_to_overlap: bool = True,
    phase_col_names: tuple[str, ...] = ('Phase', 'Δ(Phase)', 'delta Phase'),
    wrap_phase_output: bool = True,
):
    """
    Interpolate all datasets onto a common axis defined by the dataset
    with the smallest spacing.

    Only DataFrames or dict{'data','headers'} are accepted.

    Phase columns (matched by name) are unwrapped → interpolated → rewrapped.
    """
    if axis_col_name is None:
        raise ValueError(
            "axis_col_name must be provided (e.g., 'Time (ps)' or "
            "'Frequency (THz)') to ensure header-safe alignment."
        )

    objs = list(args)
    normalized = []
    axes = []
    spacings = []

    # Normalize and collect metadata
    for obj in objs:
        data, headers, fmt = _extract_data_and_headers(obj)

        if axis_col_name not in headers:
            raise KeyError(f"Axis column '{axis_col_name}' not found in headers {headers}")

        axis_idx = headers.index(axis_col_name)
        x = data[:, axis_idx]

        # Ensure x is increasing
        if len(x) > 1 and x[1] < x[0]:
            x = x[::-1]
            data = data[::-1, :]

        normalized.append((data, headers, fmt))
        axes.append(x)

        # Compute resolution
        if len(x) > 1:
            dx = np.diff(x)
            spacings.append(np.mean(np.abs(dx)))
        else:
            spacings.append(np.inf)

    # Pick the target axis = smallest spacing
    idx_best = int(np.argmin(spacings))
    x_target_full = axes[idx_best]

    # Clip to overlap region
    if clip_to_overlap:
        start = max(ax[0] for ax in axes)
        stop = min(ax[-1] for ax in axes)
        mask = (x_target_full >= start) & (x_target_full <= stop)
        x_target = x_target_full[mask]
    else:
        x_target = x_target_full

    resampled = []

    # Interpolate each dataset
    for (data, headers, fmt), x in zip(normalized, axes):

        axis_idx = headers.index(axis_col_name)
        cols = []

        for j, col_name in enumerate(headers):
            if j == axis_idx:
                cols.append(x_target)
                continue

            y = data[:, j]

            if col_name in phase_col_names:
                if is_monotonic(y):
                    # no need to unwrap monotonic data, it is already unwrapped
                    y_unwrapped = y
                else:
                    y_unwrapped = np.unwrap(y)
                y_interp = np.interp(x_target, x, y_unwrapped)
                if wrap_phase_output:
                    y_interp = np.angle(np.exp(1j * y_interp))
            else:
                y_interp = np.interp(x_target, x, y)

            cols.append(y_interp)

        new_data = np.column_stack(cols)
        resampled.append((new_data, headers, fmt))

    # Convert back to original format
    final_results = []
    for (data, headers, fmt) in resampled:
        if fmt == "dataframe":
            final_results.append(pd.DataFrame(data, columns=headers))
        else:  # fmt == 'dict'
            final_results.append({"data": data, "headers": headers})

    return final_results
