import numpy as np


def create_sequences(X, y, timestamps, input_seq_len, rolling=True):
    X_tmp, y_tmp = [], []
    ts_tmp = []
    total_length = len(y)
    # Define the threshold as a timedelta (here, in minutes)
    gap_threshold_timedelta = np.timedelta64(1, 'm')

    # Slide over the data
    i = 0
    while i < total_length - input_seq_len:
        # Get timestamps for the entire sequence (input and forecast)
        seq_timestamps = timestamps[i: i + input_seq_len]
        # Calculate differences between consecutive timestamps
        time_diffs = np.diff(seq_timestamps)
        # If any gap is larger than the allowed threshold, skip this sequence
        if np.any(time_diffs > gap_threshold_timedelta):
            if rolling:
                i += 1
            else:
                i += input_seq_len  # Skip to the end of the current sequence
            continue
        # Otherwise, create the sequence as before
        X_tmp.append(X[i: i + input_seq_len])
        y_tmp.append(y[i: i + input_seq_len])
        ts_tmp.append(seq_timestamps)
        if rolling:
            i += 1
        else:
            i += input_seq_len  # Skip to the end of the current sequence
    X, y, ts = np.array(X_tmp), np.array(y_tmp), np.array(ts_tmp)
    return X, y, ts

import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer
)

def scale_dataframe(
    df: pd.DataFrame,
    method: str = 'standard',
    columns: list[str] | None = None,
    **scaler_kwargs
) -> pd.DataFrame:
    """
    Scale numeric columns of a DataFrame using the specified method.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    method : str
        Which scaler to use. One of:
          - 'standard'  : StandardScaler (zero mean, unit variance)
          - 'minmax'   : MinMaxScaler  (scales to [0,1])
          - 'maxabs'   : MaxAbsScaler  (scales to [-1,1] by max absolute)
          - 'robust'   : RobustScaler  (uses medians and IQR)
          - 'normalize': Normalizer   (scales rows to unit norm)
    columns : list[str], optional
        List of column names to scale. If None, will scale all numeric columns.
    scaler_kwargs :
        Any additional keyword arguments for the chosen scaler.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with the specified columns scaled.
    """
    if method not in ['standard', 'minmax', 'maxabs', 'robust', 'normalize']:
        raise ValueError(f"Invalid scaling method: {method}. Choose from 'standard', 'minmax', 'maxabs', 'robust', or 'normalize'.")
    # Choose columns to scale
    if columns is None:
        cols_to_scale = df.select_dtypes(include='number').columns.tolist()
    else:
        # Only keep those that actually exist and are numeric
        existing = [c for c in columns if c in df.columns]
        num_cols = df[existing].select_dtypes(include='number').columns.tolist()
        cols_to_scale = num_cols

    if not cols_to_scale:
        # Nothing to scale
        return df.copy()

    # Select scaler
    method = method.lower()
    if method == 'standard':
        scaler = StandardScaler(**scaler_kwargs)
    elif method == 'minmax':
        scaler = MinMaxScaler(**scaler_kwargs)
    elif method == 'maxabs':
        scaler = MaxAbsScaler(**scaler_kwargs)
    elif method == 'robust':
        scaler = RobustScaler(**scaler_kwargs)
    elif method == 'normalize':
        scaler = Normalizer(**scaler_kwargs)
    else:
        raise ValueError(f"Unknown scaling method: '{method}'. "
                         "Choose from 'standard', 'minmax', 'maxabs', 'robust', 'normalize'.")

    # Fit and transform
    scaled_vals = scaler.fit_transform(df[cols_to_scale])
    scaled_df = pd.DataFrame(
        scaled_vals,
        index=df.index,
        columns=cols_to_scale
    )

    # Return a new DataFrame with scaled and untouched columns
    result = df.copy()
    result[cols_to_scale] = scaled_df

    # save scaler to file
    scaler_filename = f"{method}_scaler.pkl"
    with open(scaler_filename, 'wb') as f:
        import pickle
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_filename}")

    return result
