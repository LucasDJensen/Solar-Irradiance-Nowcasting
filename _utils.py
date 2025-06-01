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
    filename:str,
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
    if filename:
        with open(filename, 'wb') as f:
            import pickle
            pickle.dump(scaler, f)
        print(f"Scaler saved to {filename}")

    return result


import pickle
import numpy as np
import pandas as pd

def load_scaler_and_transform_df(
    filename: str,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Load a scaler (saved via pickle) and apply its transform to any columns
    in `df` that match the scaler's original feature names.

    Parameters
    ----------
    filename : str
        Path to the pickle file where the scaler was saved.
    df : pd.DataFrame
        A DataFrame whose columns you want to scale using the loaded scaler.

    Returns
    -------
    pd.DataFrame
        A copy of `df` where all columns that were seen by the loaded scaler
        have been transformed (the other columns are left unchanged).

    Raises
    ------
    ValueError
        If the loaded scaler does not have `feature_names_in_` (i.e. it wasn’t
        fit on a pandas DataFrame).
    """
    # Load scaler object
    with open(filename, 'rb') as f:
        scaler = pickle.load(f)

    # Most sklearn scalers that were fit on a DataFrame have .feature_names_in_
    if not hasattr(scaler, 'feature_names_in_'):
        raise ValueError(
            "Loaded scaler has no attribute 'feature_names_in_'. "
            "Make sure you originally fit it on a pandas DataFrame so that "
            "feature_names_in_ is set."
        )

    # Determine which of those original columns exist in the new df
    feature_names = list(scaler.feature_names_in_)
    cols_to_transform = [col for col in feature_names if col in df.columns]

    # If none of the scaler's features are in df, just return a copy
    if not cols_to_transform:
        return df.copy()

    # Apply the scaler.transform to just those columns
    scaled_vals = scaler.transform(df[cols_to_transform])
    scaled_df = pd.DataFrame(
        scaled_vals,
        index=df.index,
        columns=cols_to_transform
    )

    # Merge back into a copy of the original DataFrame
    result = df.copy()
    result[cols_to_transform] = scaled_df
    return result


def inverse_transform_targets(
    filename: str,
    df: pd.DataFrame,
    target_columns: list[str]
) -> pd.DataFrame:
    """
    Load a scaler (saved via pickle) and inverse‐transform one or more target columns.
    This assumes the scaler was originally fit on a set of columns that included these targets.

    Parameters
    ----------
    filename : str
        Path to the pickle file where the scaler was saved.
    df : pd.DataFrame
        A DataFrame containing one or more scaled target columns that you want to invert.
    target_columns : list[str]
        A list of column names (all must have been in scaler.feature_names_in_) to inverse transform.

    Returns
    -------
    pd.DataFrame
        A copy of `df` where each column in `target_columns` has been replaced by its inverse‐scaled values.
        All other columns remain unchanged.

    Raises
    ------
    ValueError
        - If the loaded scaler has no attribute `feature_names_in_`.
        - If any of `target_columns` are not in the scaler’s original feature list.
        - If the scaler type does not support inverse_transform (e.g. Normalizer).
    """
    # Load scaler object
    with open(filename, 'rb') as f:
        scaler = pickle.load(f)

    # Ensure we know the original feature names
    if not hasattr(scaler, 'feature_names_in_'):
        raise ValueError(
            "Loaded scaler has no attribute 'feature_names_in_'. "
            "Make sure it was fit on a pandas DataFrame."
        )

    feature_names = list(scaler.feature_names_in_)

    # Check that every requested target column was in the original scaler
    missing = [col for col in target_columns if col not in feature_names]
    if missing:
        raise ValueError(f"These columns were not seen by the scaler: {missing}")

    # Confirm scaler can inverse_transform
    if not hasattr(scaler, 'inverse_transform'):
        raise ValueError(
            f"The loaded scaler ({type(scaler).__name__}) does not support "
            "inverse_transform. Cannot invert these columns."
        )

    # For each target column, we extract its scaled values, build a dummy array
    # of zeros for all original features, then set the target‐column positions =
    # the scaled values. Once we call inverse_transform on that NxM array, the
    # only meaningful “reconstructed” column is our target; the rest become whatever
    # inverse_transform maps 0 → original (usually the scaler's default).
    n_samples = df.shape[0]
    n_features = len(feature_names)

    # Build an array of zeros (we'll fill in each target column one at a time)
    dummy = np.zeros((n_samples, n_features))

    # Create an output DataFrame copy
    result = df.copy()

    for tgt in target_columns:
        idx = feature_names.index(tgt)
        # Pull out the scaled column values
        scaled_vals = df[tgt].to_numpy().reshape(-1, 1)

        # Fill only that column’s position in the dummy array
        dummy[:, :] = 0.0
        dummy[:, idx] = scaled_vals[:, 0]

        # Inverse‐transform the entire dummy; then read off the idx‐th column
        inverted = scaler.inverse_transform(dummy)  # (n_samples × n_features)
        original_vals = inverted[:, idx]

        # Replace in the result DataFrame
        result[tgt] = original_vals

    return result


import pickle
import numpy as np

def inverse_transform_array(
    filename: str,
    arr: np.ndarray
) -> np.ndarray:
    """
    Load a scaler (saved via pickle) and inverse‐transform all columns of a NumPy array.

    This assumes the scaler was originally fit on an array with the same number of features
    (i.e. arr.shape[1] must match scaler.n_features_in_, if available).

    Parameters
    ----------
    filename : str
        Path to the pickle file where the scaler was saved.
    arr : np.ndarray
        A 2D NumPy array of shape (n_samples, n_features), containing scaled values
        for all features you want to invert.

    Returns
    -------
    np.ndarray
        A new NumPy array of shape (n_samples, n_features), where every column has been
        inverse‐scaled by the loaded scaler.

    Raises
    ------
    ValueError
        - If the loaded scaler does not have `n_features_in_` (so we can’t verify feature count).
        - If arr’s number of columns does not match scaler’s n_features_in_.
        - If the loaded scaler does not support `inverse_transform`.
    """
    # Load scaler object
    with open(filename, 'rb') as f:
        scaler = pickle.load(f)

    # Confirm scaler can inverse_transform
    if not hasattr(scaler, 'inverse_transform'):
        raise ValueError(
            f"The loaded scaler ({type(scaler).__name__}) does not support "
            "inverse_transform. Cannot invert the given array."
        )

    # Check that we know how many features the scaler expects
    if not hasattr(scaler, 'n_features_in_'):
        raise ValueError(
            "Loaded scaler has no attribute 'n_features_in_'. "
            "Cannot verify that the input array has the correct number of features."
        )

    expected_n = int(scaler.n_features_in_)
    idx = feature_names.index(tgt)
    # Pull out the scaled column values
    scaled_vals = df[tgt].to_numpy().reshape(-1, 1)

    # Fill only that column’s position in the dummy array
    dummy[:, :] = 0.0
    dummy[:, idx] = scaled_vals[:, 0]

    # Inverse‐transform the entire dummy; then read off the idx‐th column
    inverted = scaler.inverse_transform(dummy)  # (n_samples × n_features)
    original_vals = inverted[:, idx]

    # Replace in the result DataFrame
    result[tgt] = original_vals
    if arr.ndim != 2 or arr.shape[1] != expected_n:
        raise ValueError(
            f"Input array has shape {arr.shape}, but scaler was fit on {expected_n} features."
        )

    # Perform inverse transform on the entire array
    inverted = scaler.inverse_transform(arr)
    return inverted
