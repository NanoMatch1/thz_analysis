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
    so it can also accept a numpy.ndarray.

    - If the input arguments are numpy arrays, they are converted to DataFrames
      with the given `columns` (or those stored on the function).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            # track which args were arrays (by name)
            was_array: dict[str, bool] = {}

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
                is_arr = isinstance(value, np.ndarray)
                was_array[arg_name] = is_arr

                # Only convert *this* argument if it is an ndarray
                if is_arr:
                    df = pd.DataFrame(value, columns=list(cols))
                    bound.arguments[arg_name] = df

            # Call original function
            result = func(*bound.args, **bound.kwargs)

            # If any input was an array and output is a DataFrame, convert back
            if any(was_array.values()) and isinstance(result, pd.DataFrame):
                result = {
                    'data': result.to_numpy(),
                    'headers': result.columns.tolist()
                }

            return result

        return wrapper

    return decorator
