import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from _config import GAP_THRESHOLD
from data_loader import ProjectDataLoader
from models import Encoder, Decoder, Seq2Seq

# Initialize Weights & Biases
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
NUM_EPOCHS = 5
INPUT_SEQ_LEN = 60  # Past 60 minutes as input
FORECAST_SEQ_LEN = 60  # Forecast 60 minutes ahead
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1  # predicting one feature (GHI)
NUM_LSTM_LAYERS = 1
SPLIT = (0.65, 0.85)  # 65-85% for training; next 20% for validation; remaining 15% for test
TEACHER_FORCING_RATIO = 0.25
"""
Cumulative sum of Train/Validation/Test split.

First value is the percentage of data used for training, second value is the percentage used for validation, and the rest is used for testing.

Example: (0.6, 0.8) means 60% of the data is used for training, 20% for validation, and the remaining 20% for testing.
"""
TARGET = 'GHI'
INCLUDE_DATASET_COLUMNS = ['wind_speed_avg', 'wind_dir_avg', 'air_temperature', 'air_pressure', 'relative_humidity', 'rain_duration', 'rain_intensity', 'solar_altitude', TARGET]
wandb.init(project="solar-irradiance-nowcasting", entity="s210025-dtu",
           config={"Model": "LSTM",
                   "Dataset": "DTU Solar Station",
                   "Learning Rate": LEARNING_RATE,
                   "Batch Size": BATCH_SIZE,
                   "No. Epochs": NUM_EPOCHS,
                   "Input sequence length": INPUT_SEQ_LEN,
                   "Forecast sequence length": FORECAST_SEQ_LEN,
                   "Hidden NN layer size": HIDDEN_SIZE,
                   "Output NN layer size": OUTPUT_SIZE,
                   "No. LSTM layers": NUM_LSTM_LAYERS,
                   "Target variable": TARGET,
                   "Training variables": INCLUDE_DATASET_COLUMNS,
                   "Split": SPLIT,
                   "Teacher forcing ratio": TEACHER_FORCING_RATIO,
                   "Gap threshold": GAP_THRESHOLD
                   })

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

# Instantiate the model
input_size = data_loader.X.shape[2]  # number of features per time step

# model = LSTMEncoderDecoder(input_size, HIDDEN_SIZE, OUTPUT_SIZE, FORECAST_SEQ_LEN, NUM_LSTM_LAYERS)

# Move the model to device (GPU if available)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Instantiate the encoder, decoder, and the Seq2Seq model
encoder = Encoder(input_size, HIDDEN_SIZE, NUM_LSTM_LAYERS).to(device)
decoder = Decoder(OUTPUT_SIZE, HIDDEN_SIZE, NUM_LSTM_LAYERS).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

# -----------------------------
# 6. Training the Model with Validation
# -----------------------------
# criterion = NashSutcliffeEfficiencyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

wandb.watch(model, log="all")

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0

    with tqdm(data_loader.train_loader, unit="batch") as tepoch:
        for batch_X, batch_y in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            # Pass teacher forcing probability along with the target
            predictions = model(batch_X, target=batch_y, target_length=FORECAST_SEQ_LEN, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
            loss = criterion(predictions, batch_y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)
            tepoch.set_postfix(loss=loss.item())
    epoch_loss /= len(data_loader.train_dataset)
    wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss})

    # Optionally: Evaluate on validation set at the end of each epoch
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in data_loader.val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X, target=None, target_length=FORECAST_SEQ_LEN, teacher_forcing_ratio=0.0)  # no teacher forcing during evaluation
            loss = criterion(predictions, batch_y)
            val_loss += loss.item() * batch_X.size(0)
    val_loss /= len(data_loader.val_dataset)
    wandb.log({"epoch": epoch + 1, "val_loss": val_loss})
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

    if (epoch + 1) % 1 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
        print(f"Saved model at epoch {epoch + 1}")

wandb.finish()
