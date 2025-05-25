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