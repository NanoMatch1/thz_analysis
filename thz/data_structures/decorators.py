import functools
import inspect
from typing import Iterable, Any

import numpy as np
import pandas as pd

#TODO: Remove with_dataframes alltogether by constructing the dataframes before function call
def with_dataframe(
    columns: Iterable[str] | None = None,
    ):
    """
    Decorator to adapt a legacy function that expects a pandas.DataFrame
    so it can also accept a numpy.ndarray or dict of 'data' and 'headers'.

    - If the input arguments are numpy arrays, they are converted to DataFrames
      with the given `columns` (or those stored on the function).
    Returns:
    - If any input was an array and output is a DataFrame, converts back to dict with 'data' and 'headers'.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            # track which args were arrays (by name)
            format_type: dict[str, bool] = {}

            # Determine columns: decorator > function attribute
            cols = columns
            if cols is None:
                cols = getattr(func, "_expected_columns", None)
            if cols is None:
                raise ValueError(
                    f"No column labels provided for {func.__name__}. "
                    f"Pass 'columns=...' to the decorator or set "
                    f"func._expected_columns = [...]"
                )

            for arg_name, value in bound.arguments.items():
                if isinstance(value, pd.DataFrame):
                    format_type[arg_name] = "dataframe"
                    continue
                is_arr = isinstance(value, np.ndarray)
                is_dict = isinstance(value, dict) and 'data' in value and 'headers' in value
                if not (is_arr or is_dict):
                    format_type[arg_name] = "other"
                    continue
                format_type[arg_name] = "array" if is_arr else "dict"



                # Only convert *this* argument if it is an ndarray
                if is_arr:
                    df = pd.DataFrame(value, columns=list(cols))
                    bound.arguments[arg_name] = df
                elif is_dict:
                    df = pd.DataFrame(value['data'], columns=value['headers'])
                    bound.arguments[arg_name] = df

            # Call original function
            result = func(*bound.args, **bound.kwargs)

            # If any input was an array and output is a DataFrame, convert back
            if any(ft in ("array", "dict") for ft in format_type.values()) and isinstance(result, pd.DataFrame):
                result = {
                    'data': result.to_numpy(),
                    'headers': result.columns.tolist()
                }

            return result

        return wrapper

    return decorator
