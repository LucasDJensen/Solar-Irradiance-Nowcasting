import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from _config import GAP_THRESHOLD
from data_loader import ProjectDataLoader
from models import Encoder, Decoder, Seq2Seq

# -----------------------------
# 1. Parameters and Hyperparameters
# -----------------------------
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 5
INPUT_SEQ_LEN = 60  # Past 60 minutes as input
FORECAST_SEQ_LEN = 60  # Forecast 60 minutes ahead
HIDDEN_SIZE = 128
OUTPUT_SIZE = 1  # predicting one feature (GHI)
NUM_LSTM_LAYERS = 2
SPLIT = (0.65, 0.85)  # 65-85% for training; next 20% for validation; remaining 15% for test
TEACHER_FORCING_RATIO = 0.25
"""
Cumulative sum of Train/Validation/Test split.

First value is the percentage of data used for training, second value is the percentage used for validation, and the rest is used for testing.

Example: (0.6, 0.8) means 60% of the data is used for training, 20% for validation, and the remaining 20% for testing.
"""
TARGET = 'GHI'
INCLUDE_DATASET_COLUMNS = ['wind_speed_avg', 'wind_dir_avg', 'air_temperature', 'air_pressure', 'relative_humidity', 'rain_duration', 'rain_intensity', 'solar_altitude', TARGET]

# -----------------------------
# 2. Load the scalers (for inverse transformation if needed)
# -----------------------------
with open("scalers.pkl", "rb") as f:
    scalers = pickle.load(f)
feature_scaler = scalers["feature_scaler"]
target_scaler = scalers["target_scaler"]

# -----------------------------
# 1. Data Loading and Preprocessing
# -----------------------------
data_loader = ProjectDataLoader(TARGET, INCLUDE_DATASET_COLUMNS, INPUT_SEQ_LEN, FORECAST_SEQ_LEN, GAP_THRESHOLD, SPLIT)
data_loader.load_data()
data_loader.transform_data()
# -----------------------------
# 2. Creating Sequences for Supervised Learning
# -----------------------------
data_loader.create_sequences(data_loader.data)

print("X shape:", data_loader.X.shape)
print("y shape:", data_loader.y.shape)
# -----------------------------
# 3. Chronological Train/Val/Test Split
# -----------------------------
# -----------------------------
# 4. Creating a PyTorch Dataset and DataLoaders
# -----------------------------
data_loader.init_pytorch_datasets_and_loaders(BATCH_SIZE)
# -----------------------------
# 5. Defining the LSTM Encoder-Decoder Model in PyTorch
# -----------------------------
input_size = data_loader.X.shape[2]  # number of features per time step

# model = LSTMEncoderDecoder(input_size, HIDDEN_SIZE, OUTPUT_SIZE, FORECAST_SEQ_LEN, NUM_LSTM_LAYERS)
# Move the model to device (GPU if available)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# model.to(device)

encoder = Encoder(input_size, HIDDEN_SIZE, NUM_LSTM_LAYERS).to(device)
decoder = Decoder(OUTPUT_SIZE, HIDDEN_SIZE, NUM_LSTM_LAYERS).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

# -----------------------------
# 8. Load the Trained Model Checkpoint
# -----------------------------
# Specify the checkpoint file (change if needed)
checkpoint_path = "model_epoch_1.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded model checkpoint from {checkpoint_path}")
else:
    raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")

# -----------------------------
# 9. Evaluate the Model on the Test Set
# -----------------------------
model.eval()
# criterion = NashSutcliffeEfficiencyLoss()
criterion = nn.MSELoss()
test_loss = 0.0
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_y in tqdm(data_loader.test_loader, desc="Evaluating"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        predictions = model(batch_X, target=None, target_length=FORECAST_SEQ_LEN, teacher_forcing_ratio=0.0)
        loss = criterion(predictions, batch_y)
        test_loss += loss.item() * batch_X.size(0)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

test_loss /= len(data_loader.test_dataset)
print(f"Test Loss (MSE): {test_loss:.6f}")

# Concatenate predictions and targets
all_predictions = np.concatenate(all_predictions, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# -----------------------------
# 10. Plotting: Compare Predictions and Ground Truth for Sample Sequences
# -----------------------------
# Plot a few random samples from the test set
num_samples_to_plot = 10
indices = np.random.choice(all_predictions.shape[0], num_samples_to_plot, replace=False)

for i, idx in enumerate(indices):
    pred_seq = all_predictions[idx].squeeze()  # shape: [forecast_seq_len]
    true_seq = all_targets[idx].squeeze()

    plt.figure(figsize=(10, 4))
    plt.plot(pred_seq, label="Prediction", linestyle="--")
    plt.plot(true_seq, label="Ground Truth", linestyle="-")
    plt.title(f"Test Sample {idx}")
    plt.xlabel("Time step")
    plt.ylabel("Scaled GHI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"test_sample_{i}_comparison.png")
    plt.show()

# -----------------------------
# 11. Plotting: Error Distribution
# -----------------------------
errors = all_predictions - all_targets
plt.figure(figsize=(8, 4))
plt.hist(errors.flatten(), bins=50, alpha=0.7)
plt.title("Error Distribution (Prediction - Ground Truth)")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.tight_layout()
# plt.savefig("error_distribution.png")
plt.show()
