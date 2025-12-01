import pandas as pd
import numpy as np

def df_to_array(df: pd.DataFrame) -> np.ndarray:
    """Convert a pandas DataFrame to a numpy ndarray - columns not preserved."""
    return df.to_numpy()

def df_to_dict(df: pd.DataFrame) -> dict:
    """Convert a pandas DataFrame to a dictionary of numpy ndarrays."""
    return {col: df[col].to_numpy() for col in df.columns}

def dict_to_df(data_dict: dict) -> pd.DataFrame:
    """Convert a dictionary of numpy ndarrays to a pandas DataFrame."""
    return pd.DataFrame(data_dict)

def array_to_df(array: np.ndarray, columns: list[str] | None = None) -> pd.DataFrame:
    """Convert a numpy ndarray to a pandas DataFrame."""
    return pd.DataFrame(array, columns=columns)