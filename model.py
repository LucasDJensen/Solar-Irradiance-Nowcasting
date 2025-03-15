import torch

from _config import GAP_THRESHOLD
from data_loader import ProjectDataLoader
from models import SimpleLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGETS = ['DNI', 'DHI']
INCLUDE_DATASET_COLUMNS = ['wind_speed_avg', 'wind_dir_avg', 'air_temperature', 'air_pressure', 'relative_humidity', 'rain_intensity', 'solar_altitude']
INPUT_SEQ_LEN = 60  # Past 60 minutes as input
FORECAST_SEQ_LEN = 60  # Forecast 60 minutes ahead

NUM_LSTM_LAYERS = 2
HIDDEN_SIZE = 64
OUTPUT_SIZE = len(TARGETS)  # predicting two features (DNI, DHI)

BATCH_SIZE = 32
SPLIT = (0.65, 0.85)  # 65-85% for training; next 20% for validation; remaining 15% for test
"""
Cumulative sum of Train/Validation/Test split.

First value is the percentage of data used for training, second value is the percentage used for validation, and the rest is used for testing.

Example: (0.6, 0.8) means 60% of the data is used for training, 20% for validation, and the remaining 20% for testing.
"""

# -----------------------------
# 3. Data Loading and Preprocessing
# -----------------------------
data_loader = ProjectDataLoader(TARGETS, INCLUDE_DATASET_COLUMNS, INPUT_SEQ_LEN, FORECAST_SEQ_LEN, GAP_THRESHOLD, SPLIT, device=DEVICE)
data_loader.load_data()
# data_loader.feature_engineering()
# data_loader.transform_data()
print("X shape:", data_loader.X.shape)
print("y shape:", data_loader.y.shape)
# -----------------------------
# 4. Creating Sequences for Supervised Learning
# -----------------------------
data_loader.create_sequences()
data_loader.init_pytorch_datasets_and_loaders(BATCH_SIZE)

input_size = data_loader.X.shape[2]  # number of features per time step
model = SimpleLSTM(input_size, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LSTM_LAYERS).to(DEVICE)

