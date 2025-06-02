
import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        X_full: np.ndarray,           # shape = (T, num_features), dtype should be float32
        y_full: np.ndarray,           # shape = (T, num_targets),  dtype float32
        ts_full: np.ndarray,          # shape = (T,),             dtype datetime64[ns] (or similar)
        input_seq_len: int,
        gap_threshold_minutes: int = 1
    ):
        """
        A sliding-window timeseries Dataset that *skips* any window
        whose consecutive timestamps differ by more than `gap_threshold_minutes`.

        Parameters
        ----------
        X_full : np.ndarray
            2D array of shape (T, num_features), dtype float32.
        y_full : np.ndarray
            2D array of shape (T, num_targets), dtype float32.
        ts_full : np.ndarray
            1D array of shape (T,), dtype datetime64 (e.g. np.datetime64[ns]).
        input_seq_len : int
            Number of timesteps per input window.
        gap_threshold_minutes : int
            Any jump between two consecutive timestamps > this (in minutes)
            invalidates the entire window.
        """
        assert X_full.shape[0] == y_full.shape[0] == ts_full.shape[0], \
            "X, y, and ts must all have the same first dimension."

        self.seq_len = input_seq_len

        # Cast features/targets to float32 immediately (if not already)
        if X_full.dtype != np.float32:
            X_full = X_full.astype(np.float32)
        if y_full.dtype != np.float32:
            y_full = y_full.astype(np.float32)

        self.X_full = X_full
        self.y_full = y_full
        self.ts_full = ts_full

        T = ts_full.shape[0]
        if T < input_seq_len:
            raise ValueError(
                f"Not enough total timesteps ({T}) for sequence length {input_seq_len}"
            )

        # Compute diffs between consecutive timestamps:
        #   diffs[i] = ts_full[i+1] - ts_full[i], length = T - 1
        diffs = np.diff(ts_full)  # dtype = timedelta64[ns] (or similar)

        # We will mark any window starting at i as invalid if
        # any diffs[i : i+seq_len-1] > gap_threshold.
        gap_thresh = np.timedelta64(gap_threshold_minutes, 'm')

        valid_starts = []
        last_start = T - input_seq_len  # inclusive

        # Iterate over all possible start‐indices i = 0..(T−seq_len)
        for i in range(last_start + 1):
            # Look at diffs from i .. (i + seq_len − 2) inclusive; that's
            # exactly (seq_len−1) differences for a seq of length `seq_len`.
            window_diffs = diffs[i : i + (input_seq_len - 1)]
            if not np.any(window_diffs > gap_thresh):
                valid_starts.append(i)

        if len(valid_starts) == 0:
            raise ValueError(
                f"No valid sliding windows of length {input_seq_len} found "
                "given the timestamp gaps and threshold."
            )

        # Turn into a NumPy array of ints once.
        self.valid_start_idxs = np.array(valid_starts, dtype=np.int64)

    def __len__(self):
        # Number of valid windows after rejecting any with a gap > threshold
        return len(self.valid_start_idxs)

    def __getitem__(self, idx: int):
        """
        Returns a single (X_window, y_label) pair:
          - X_window: tensor of shape (seq_len, num_features), float32
          - y_label : tensor of shape (seq_len,), float32
          - ts_window: numpy array of shape (seq_len,), dtype datetime64[ns] (or similar)
        """
        # idx is an index into the list of valid starts
        start = int(self.valid_start_idxs[idx])
        end = start + self.seq_len  # exclusive

        # Extract feature‐slice and the “last‐timestep” target
        x_win = self.X_full[start:end, :]          # shape = (seq_len, num_features)
        y_win = self.y_full[start:end, :]             # shape = (num_targets,)

        # Convert to torch.Tensor
        x_tensor = torch.from_numpy(x_win)          # dtype=torch.float32
        y_tensor = torch.from_numpy(y_win)          # dtype=torch.float32

        return x_tensor, y_tensor, idx

