import numpy as np


def create_sequences(X, y, timestamps, input_seq_len, forecast_seq_len, gap_threshold=10):
    """
    Generates sliding windows for sequence-to-sequence learning.
    Uses 'input_seq_len' minutes of past data to predict the next
    'forecast_seq_len' values for the target feature.

    Parameters:
      data_array: NumPy array of shape (n_samples, n_features)
      timestamps: NumPy array of datetime64 values corresponding to each row in data_array
      input_seq_len: int, number of time steps for input
      forecast_seq_len: int, number of time steps for forecasting
      gap_threshold: int, maximum allowed gap (in minutes) between consecutive time stamps
      target_col_index: int, index of the target column in data_array

    Returns:
      X: NumPy array of input sequences
      y: NumPy array of corresponding target sequences
    """
    X_tmp, y_tmp = [], []
    total_length = len(y)
    # Define the threshold as a timedelta (here, in minutes)
    gap_threshold_timedelta = np.timedelta64(gap_threshold, 'm')

    # Slide over the data
    for i in range(total_length - input_seq_len - forecast_seq_len + 1):
        # Get timestamps for the entire sequence (input and forecast)
        seq_timestamps = timestamps[i: i + input_seq_len + forecast_seq_len]
        # Calculate differences between consecutive timestamps
        time_diffs = np.diff(seq_timestamps)
        # If any gap is larger than the allowed threshold, skip this sequence
        if np.any(time_diffs > gap_threshold_timedelta):
            continue
        # Otherwise, create the sequence as before
        X_tmp.append(X[i: i + input_seq_len])
        y_tmp.append(y[i + input_seq_len: i + input_seq_len + forecast_seq_len])
    return np.array(X_tmp), np.array(y_tmp)

