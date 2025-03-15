import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset

from _config import FILE_PROCESSED_DTU_SOLAR_STATION, FILE_SCALARS
from sequencing import create_sequences


class ProjectDataLoader:
    X: pd.DataFrame = None
    y: pd.DataFrame = None

    train_dataset: Subset = None
    val_dataset: Subset = None
    test_dataset: Subset = None

    train_loader: DataLoader = None
    val_loader: DataLoader = None
    test_loader: DataLoader = None
    device: str

    def __init__(self, target, include_dataset_columns, input_seq_len, forecast_seq_len, gap_threshold, split, device='cpu'):
        """
        Initializes the DataProcessor with the required parameters.

        Parameters:
            target (list): The target column name.
            include_dataset_columns (list): List of dataset columns to include.
            input_seq_len (int): Length of the input sequence.
            forecast_seq_len (int): Length of the forecast sequence.
            gap_threshold (float or int): The gap threshold value used in sequence creation.
            split (tuple): A tuple (train_split, val_split) where each value is a fraction of total samples.
                           For example, (0.65, 0.85) means 65% for training, next 20% for validation, rest for test.
        """
        self.target = target
        self.include_dataset_columns = include_dataset_columns
        self.input_seq_len = input_seq_len
        self.forecast_seq_len = forecast_seq_len
        self.gap_threshold = gap_threshold
        self.split = split
        self.device = device

    def load_data(self):
        """Loads and preprocesses the dataset using a custom loader."""
        with open(FILE_PROCESSED_DTU_SOLAR_STATION, 'rb') as f:
            data: pd.DataFrame = pickle.load(f)

        data = data.loc['2022-01-01']

        self.X = data.drop(columns=self.target, axis=1)
        self.y = data[self.target]

    def feature_engineering(self, num_lags=60):
        for i in range(1, num_lags + 1):
            column_name = f'y_-{i}t'
            self.X[column_name] = self.y.shift(-i)
            self.include_dataset_columns.append(column_name)

    def transform_data(self):
        # Scale features using StandardScaler
        feature_scaler = StandardScaler()
        scaled_features = feature_scaler.fit_transform(self.X)

        # Scale target using MinMaxScaler
        target_scaler = MinMaxScaler()
        # Reshape the target to 2D as expected by MinMaxScaler
        scaled_target = target_scaler.fit_transform(self.y.values.reshape(-1, 1))

        # (Optional) Create DataFrames from the scaled arrays
        scaled_features_df = pd.DataFrame(scaled_features, columns=self.X.columns, index=self.X.index)
        scaled_target_df = pd.DataFrame(scaled_target, columns=self.target, index=self.y.index)
        self.X = scaled_features_df
        self.y = scaled_target_df

        # Store scalers for inverse transformations using pickle
        scalers = {"feature_scaler": feature_scaler, "target_scaler": target_scaler}
        with open(FILE_SCALARS, "wb") as f:
            pickle.dump(scalers, f)

    def create_sequences(self):
        """
        Creates sliding window sequences for supervised learning and reshapes the target variable.
        Uses `input_seq_len` minutes of past data to predict the next `forecast_seq_len` values.

        Assumes that:
          - self.X is a pandas DataFrame with a DateTimeIndex.
          - self.y is the corresponding target variable.
          - self.input_seq_len, self.forecast_seq_len, and self.gap_threshold are set.
        """
        # Extract timestamps from the index (assumes a datetime index)
        timestamps = self.X.index.to_numpy()

        # Convert X and y to NumPy arrays
        X_array = self.X.to_numpy()
        y_array = self.y.to_numpy()

        X_sequences, y_sequences = [], []
        total_length = len(y_array)

        # Define the threshold as a timedelta (in minutes)
        gap_threshold_timedelta = np.timedelta64(self.gap_threshold, 'm')

        # Generate sliding window sequences
        for i in range(total_length - self.input_seq_len - self.forecast_seq_len + 1):
            # Get timestamps for the entire sequence (input + forecast)
            seq_timestamps = timestamps[i: i + self.input_seq_len + self.forecast_seq_len]
            # Calculate differences between consecutive timestamps
            if np.any(np.diff(seq_timestamps) > gap_threshold_timedelta):
                continue  # Skip this sequence if any gap is too large

            # Append input sequence and forecast sequence
            X_sequences.append(X_array[i: i + self.input_seq_len])
            y_sequences.append(y_array[i + self.input_seq_len: i + self.input_seq_len + self.forecast_seq_len])

        # Update the instance variables with the new sequences.
        self.X = np.array(X_sequences)
        self.y = np.array(y_sequences).reshape(-1, self.forecast_seq_len, len(self.target))


    def split_data(self):
        """
        Splits the dataset indices into training, validation, and test sets based on the chronological order.

        Returns:
            tuple: (train_indices, val_indices, test_indices)
        """
        # Ensure sequences are created
        if self.X is None:
            raise ValueError("Sequences not created. Call create_sequences() first.")

        total_samples = self.X.shape[0]
        train_end = int(self.split[0] * total_samples)  # e.g., 65% for training
        val_end = int(self.split[1] * total_samples)  # e.g., next 20% for validation

        train_indices = list(range(0, train_end))
        val_indices = list(range(train_end, val_end))
        test_indices = list(range(val_end, total_samples))

        return train_indices, val_indices, test_indices

    def init_pytorch_datasets_and_loaders(self, batch_size):
        dataset = TimeSeriesDataset(self.X, self.y, device=self.device)
        train_indices, val_indices, test_indices = self.split_data()
        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)
        self.test_dataset = Subset(dataset, test_indices)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, device='cpu'):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)  # [samples, seq_len, features]
        self.y = torch.tensor(y, dtype=torch.float32).to(device)  # [samples, forecast_seq_len, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
