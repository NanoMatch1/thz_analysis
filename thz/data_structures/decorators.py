import functools
import inspect
from typing import Iterable, Any

import numpy as np
import pandas as pd


def array_to_dataframe_adapter(
    arg_name: str = "timedata",
    columns: Iterable[str] | None = None,
):
    """
    Decorator to adapt a legacy function that expects a pandas.DataFrame
    so it can also accept a numpy.ndarray.

    - If `arg_name` is a numpy array, it is converted to a DataFrame
      with the given `columns` (or those stored on the function).
    - After the function call, if the original input was an array,
      the (possibly modified) DataFrame is converted back to a numpy array
      and returned.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            if arg_name not in bound.arguments:
                raise TypeError(
                    f"{func.__name__}() missing required argument '{arg_name}' "
                    f"for array_to_dataframe_adapter"
                )

            value = bound.arguments[arg_name]
            was_array = isinstance(value, np.ndarray)

            # Determine columns: decorator > function attribute
            cols = columns
            if cols is None:
                cols = getattr(func, "_expected_columns", None)

            if was_array:
                if cols is None:
                    raise ValueError(
                        f"No column labels provided for {func.__name__}. "
                        f"Pass 'columns=...' to the decorator or set "
                        f"func._expected_columns = [...]"
                    )
                df = pd.DataFrame(value, columns=list(cols))
                bound.arguments[arg_name] = df

            # Call original function
            result = func(*bound.args, **bound.kwargs)

            # If original input was an array, convert back
            if was_array:
                df_now = bound.arguments[arg_name]

                # If the function returns a DataFrame, prefer that
                if isinstance(result, pd.DataFrame):
                    return result.to_numpy()

                # If it returns None and works in-place, use the mutated df
                if result is None and isinstance(df_now, pd.DataFrame):
                    return df_now.to_numpy()

                # Otherwise, just return whatever it returned
                return result

            # If input was already a DataFrame, do nothing special
            return result

        return wrapper

    return decorator
