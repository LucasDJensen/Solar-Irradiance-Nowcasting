import numpy as np
import torch
from torch.utils.data import Dataset

class TimeSeriesDatasetGPU(Dataset):
    def __init__(
        self,
        X_full: np.ndarray,
        y_full: np.ndarray,
        ts_full: np.ndarray,
        input_seq_len: int,
        gap_threshold_minutes: int = 1,
        device: torch.device = torch.device("cpu")
    ):
        """
        A sliding‐window timeseries Dataset that:
          1) Finds all valid windows on CPU (same as before).
          2) Moves the entire X_full/y_full arrays to `device` (GPU) as torch.float32.
          3) __getitem__ simply slices the GPU tensors.

        Parameters
        ----------
        X_full : np.ndarray
            2D array of shape (T, num_features), dtype float32 (or casted).
        y_full : np.ndarray
            2D array of shape (T, num_targets), dtype float32 (or casted).
        ts_full : np.ndarray
            1D array of shape (T,), dtype datetime64[ns].
        input_seq_len : int
            Number of timesteps per input window.
        gap_threshold_minutes : int
            Any jump > this threshold invalidates the window.
        device : torch.device
            The device (e.g. torch.device("cuda")) onto which we put X_full and y_full.
        """
        assert X_full.shape[0] == y_full.shape[0] == ts_full.shape[0], \
            "X, y, and ts must have the same length."

        self.seq_len = input_seq_len
        self.device = device

        # 1) Cast to float32 if needed (still on CPU)
        if X_full.dtype != np.float32:
            X_full = X_full.astype(np.float32)
        if y_full.dtype != np.float32:
            y_full = y_full.astype(np.float32)

        # 2) Compute valid window start indices on CPU (exactly as before)
        T = ts_full.shape[0]
        if T < input_seq_len:
            raise ValueError(
                f"Not enough timesteps ({T}) for sequence length {input_seq_len}"
            )

        diffs = np.diff(ts_full)  # dtype: timedelta64
        gap_thresh = np.timedelta64(gap_threshold_minutes, 'm')

        valid_starts = []
        last_start = T - input_seq_len  # inclusive
        for i in range(last_start + 1):
            window_diffs = diffs[i : i + (input_seq_len - 1)]
            if not np.any(window_diffs > gap_thresh):
                valid_starts.append(i)

        if len(valid_starts) == 0:
            raise ValueError(
                f"No valid sliding windows of length {input_seq_len} "
                "found given the timestamp gaps and threshold."
            )

        self.valid_start_idxs = np.array(valid_starts, dtype=np.int64)

        # 3) Now move X_full, y_full to GPU as torch.Tensor (once)
        #    Non‐blocking is okay because these NumPy → Tensor → GPU copies happen in __init__, not per‐batch.
        #    After this, self.X_full_t and self.y_full_t live on GPU.
        self.X_full_t = torch.from_numpy(X_full).to(device, non_blocking=True)
        self.y_full_t = torch.from_numpy(y_full).to(device, non_blocking=True)
        self.ts_full = ts_full

        # (No need to keep ts_full around—indexing & gap logic is done.)

    def __len__(self):
        return len(self.valid_start_idxs)

    def __getitem__(self, idx: int):
        """
        Returns a single (X_window, y_window) pair,
        both already on GPU.

        - X_window:      torch.Tensor, shape = (seq_len, num_features), dtype=torch.float32
        - y_window:      torch.Tensor, shape = (seq_len, num_targets),   dtype=torch.float32
        """
        start = int(self.valid_start_idxs[idx])
        end = start + self.seq_len

        # Because X_full_t and y_full_t are already on GPU, slicing returns GPU tensors.
        x_win = self.X_full_t[start:end, :]  # shape = (seq_len, num_features), on GPU
        y_win = self.y_full_t[start:end, :]  # shape = (seq_len, num_targets), on GPU

        return x_win, y_win, idx
