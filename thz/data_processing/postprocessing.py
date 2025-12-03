import numpy as np
import pandas as pd
import functools


def _extract_data_and_headers(obj):
    """
    Normalize input to (data, headers, fmt).

    fmt is one of: 'dataframe', 'ndarray', 'dict'.
    Data is always returned as a 2D ndarray.
    """
    if isinstance(obj, pd.DataFrame):
        data = obj.to_numpy()
        headers = obj.columns.tolist()
        fmt = "dataframe"
    elif isinstance(obj, np.ndarray):
        data = obj
        headers = None
        fmt = "ndarray"
    elif isinstance(obj, dict) and "data" in obj and "headers" in obj:
        data = obj["data"]
        headers = obj["headers"]
        fmt = "dict"
    else:
        raise TypeError(f"Unsupported data type: {type(obj)}")

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return data, headers, fmt


def interpolate_to_max_resolution(
    *args,
    axis_col: int = 0,
    clip_to_overlap: bool = True,
    phase_col_names: tuple[str, ...] = ("Phase",),   # phase columns by name
    phase_col_indices: tuple[int, ...] = (),         # phase columns by index (for ndarrays)
    wrap_phase_output: bool = True,
):
    """
    Interpolate multiple datasets onto the axis of the dataset with the
    finest resolution (smallest mean spacing in axis_col).

    Supports:
      - pandas.DataFrame
      - numpy.ndarray
      - dict {'data': ndarray, 'headers': list}

    Phase columns:
      - For DataFrames/dicts: specify by name via `phase_col_names`.
      - For ndarrays (no headers): specify by index via `phase_col_indices`.

    Phase columns are:
      - unwrapped with np.unwrap,
      - interpolated on the continuous phase,
      - optionally re-wrapped to [-π, π] if wrap_phase_output=True.
    """
    objs = list(args)
    normalized = []
    axes = []
    spacings = []

    # Normalize inputs
    for obj in objs:
        data, headers, fmt = _extract_data_and_headers(obj)

        x = data[:, axis_col]
        # Ensure axis is increasing
        if len(x) > 1 and x[1] < x[0]:
            x = x[::-1]
            data = data[::-1, :]

        normalized.append((data, headers, fmt))
        axes.append(x)

        if len(x) > 1:
            dx = np.diff(x)
            spacings.append(np.mean(np.abs(dx)))
        else:
            spacings.append(np.inf)

    # Choose target axis = axis of dataset with smallest spacing
    idx_best = int(np.argmin(spacings))
    x_target_full = axes[idx_best]

    # Overlap region
    if clip_to_overlap:
        start = max(ax[0] for ax in axes)
        stop = min(ax[-1] for ax in axes)
        mask = (x_target_full >= start) & (x_target_full <= stop)
        x_target = x_target_full[mask]
    else:
        x_target = x_target_full

    resampled = []

    for (data, headers, fmt), x in zip(normalized, axes):
        cols = []

        # Build a mapping from header name to index (if headers exist)
        name_to_idx = {}
        if headers is not None:
            name_to_idx = {name: idx for idx, name in enumerate(headers)}

        for j in range(data.shape[1]):
            if j == axis_col:
                # axis column → target axis
                cols.append(x_target)
                continue

            # Determine if this column is a phase column
            is_phase = False
            if headers is not None:
                col_name = headers[j]
                if col_name in phase_col_names:
                    is_phase = True
            else:
                # ndarray with no headers: use indices
                if j in phase_col_indices:
                    is_phase = True

            y = data[:, j]

            if is_phase:
                # unwrap → interp → rewrap (optional)
                y_unwrapped = np.unwrap(y)
                y_interp_unwrapped = np.interp(x_target, x, y_unwrapped)
                if wrap_phase_output:
                    y_interp = np.angle(np.exp(1j * y_interp_unwrapped))
                else:
                    y_interp = y_interp_unwrapped
            else:
                # normal scalar interpolation
                y_interp = np.interp(x_target, x, y)

            cols.append(y_interp)

        new_data = np.column_stack(cols)
        resampled.append((new_data, headers, fmt))

    # Convert back to original formats
    final_results = []
    for (data, headers, fmt), orig_obj in zip(resampled, objs):
        if fmt == "dataframe":
            df = pd.DataFrame(data, columns=headers)
            final_results.append(df)
        elif fmt == "ndarray":
            final_results.append(data)
        elif fmt == "dict":
            final_results.append({"data": data, "headers": headers})

    return final_results
