import os
import pickle

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from _config import PATH_CHECKPOINT, FILE_SCALARS
from model import data_loader, model, DEVICE

with open(FILE_SCALARS, "rb") as f:
    scaler: MinMaxScaler = pickle.load(f)

# -----------------------------
# 8. Load the Trained Model Checkpoint
# -----------------------------
# Find all checkpoint files in the directory following the pattern checkpoint_{epoch}.pt
checkpoint_files = [f for f in os.listdir(PATH_CHECKPOINT) if f.startswith('checkpoint_') and f.endswith('.pt')]
if checkpoint_files:
    # Sort files by epoch number in descending order and choose the latest checkpoint.
    checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
    latest_checkpoint_file = checkpoint_files[0]

    model.load_state_dict(torch.load(PATH_CHECKPOINT / latest_checkpoint_file, map_location=DEVICE)['model_state_dict'])
else:
    raise FileNotFoundError(f"No checkpoint files found in {PATH_CHECKPOINT}")

# -----------------------------
# 9. Evaluate the Model on the Test Set
# -----------------------------
model.eval()
criterion = nn.SmoothL1Loss()
test_loss = 0.0
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_y in tqdm(data_loader.test_loader, desc="Evaluating"):
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        predictions = model(batch_X)  # , target=None, target_length=FORECAST_SEQ_LEN, teacher_forcing_ratio=0.0)
        loss = criterion(predictions, batch_y)
        test_loss += loss.item() * batch_X.size(0)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

test_loss /= len(data_loader.test_dataset)
print(f"Test Loss (MSE): {test_loss:.6f}")

# Concatenate predictions and targets
predictions = all_predictions
targets = all_targets




# -----------------------------
# 10. Plotting: Compare Predictions and Ground Truth for Sample Sequences
# -----------------------------
