from sklearn.preprocessing import MinMaxScaler

TARGETS = ['DNI', 'DHI']
# INCLUDE_DATASET_COLUMNS = ['wind_speed_avg', 'wind_dir_avg', 'air_temperature', 'air_pressure', 'relative_humidity', 'rain_intensity', 'solar_altitude']
INPUT_SEQ_LEN = 60  # Past 60 minutes as input
FORECAST_SEQ_LEN = 60  # Forecast 60 minutes ahead

NUM_LSTM_LAYERS = 3
HIDDEN_SIZE = 128
OUTPUT_SIZE = len(TARGETS)  # predicting two features (DNI, DHI)

BATCH_SIZE = 128
SPLIT = (0.65, 0.85)  # 65-85% for training; next 20% for validation; remaining 15% for test
"""
Cumulative sum of Train/Validation/Test split.

First value is the percentage of data used for training, second value is the percentage used for validation, and the rest is used for testing.

Example: (0.6, 0.8) means 60% of the data is used for training, 20% for validation, and the remaining 20% for testing.
"""


def main():
    import os

    import pandas as pd
    import torch.nn as nn
    import wandb
    import numpy as np
    from tqdm import tqdm

    from ForecastEvaluator import ForecastEvaluator  # Utility class for evaluation metrics
    from _config import GAP_THRESHOLD, PATH_CHECKPOINT, PKL_PROCESSED_STEP2_DTU_SOLAR_STATION
    from models import SimpleLSTM
    import torch
    from torch.utils.data import Dataset, DataLoader, Subset

    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y, device):
            self.X = torch.tensor(X, dtype=torch.float32).to(device)  # [samples, seq_len, features]
            self.y = torch.tensor(y, dtype=torch.float32).to(device)  # [samples, forecast_seq_len, 1]

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # -----------------------------
    # 3. Data Loading and Preprocessing
    # -----------------------------
    df: pd.DataFrame = pd.read_pickle(PKL_PROCESSED_STEP2_DTU_SOLAR_STATION)
    df.dropna(inplace=True)
    # scale
    minmax_scaler = MinMaxScaler()
    features = [x for x in df.columns if x not in TARGETS]
    df[features] = minmax_scaler.fit_transform(df[features])

    X = df.drop(columns=TARGETS)
    y = df[TARGETS]
    # -----------------------------
    # 4. Creating Sequences for Supervised Learning
    # -----------------------------
    total_samples = X.shape[0]
    train_end = int(SPLIT[0] * total_samples)  # e.g., 65% for training
    val_end = int(SPLIT[1] * total_samples)  # e.g., next 20% for validation

    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, total_samples))

    dataset = TimeSeriesDataset(X.to_numpy(), y.to_numpy(), device=device)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_size = X.shape[1]  # number of features per time step
    model = SimpleLSTM(input_size, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LSTM_LAYERS).to(device)

    # -----------------------------
    # 1. Parameters and Hyperparameters
    # -----------------------------
    # Set to True if you want to resume training from a checkpoint.
    RESUME = False

    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 200
    # TEACHER_FORCING_RATIO = 0.7
    SCHEDULER_PATIENCE = 5

    # -----------------------------
    # 6. Model, Loss, Optimizer, and Scheduler Setup
    # -----------------------------
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Define the scheduler (this will reduce the LR if the validation loss does not improve)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=SCHEDULER_PATIENCE)

    # -----------------------------
    # 2. Initialize wandb with resume support
    # -----------------------------
    resume_flag = "allow" if RESUME else "never"
    wandb.init(project="solar-irradiance-nowcasting", entity="s210025-dtu", resume=resume_flag,
               config={"Dataset": "DTU Solar Station",
                       "Model": "SimpleLSTM",
                       "Input Sequence Length": INPUT_SEQ_LEN,
                       "Forecast Sequence Length": FORECAST_SEQ_LEN,
                       "Hidden Size": HIDDEN_SIZE,
                       "LSTM Layers": NUM_LSTM_LAYERS,
                       "Batch Size": BATCH_SIZE,
                       "Learning Rate": LEARNING_RATE,
                       # "Teacher Forcing Ratio": TEACHER_FORCING_RATIO,
                       "Scheduler Patience": SCHEDULER_PATIENCE,
                       "Split": SPLIT,
                       "Gap Threshold": GAP_THRESHOLD,
                       "Targets": TARGETS,
                       "Output Size": OUTPUT_SIZE,
                       "Loss": criterion,
                       "Optimizer": optimizer,
                       "Scheduler": scheduler,
                       "Dropout": 0.5,
                       })

    wandb.watch(model, log="all")

    # -----------------------------
    # 7. Resume Training Setup (for model & optimizer)
    # -----------------------------
    start_epoch = 0

    if RESUME:
        # Find all checkpoint files in the directory following the pattern checkpoint_{epoch}.pt
        checkpoint_files = [f for f in os.listdir(PATH_CHECKPOINT) if f.startswith('checkpoint_') and f.endswith('.pt')]
        if checkpoint_files:
            # Sort files by epoch number in descending order and choose the latest checkpoint.
            checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            latest_checkpoint_file = checkpoint_files[0]

            checkpoint = torch.load(PATH_CHECKPOINT / latest_checkpoint_file, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

            print(f"Resuming training from epoch {start_epoch} using checkpoint {latest_checkpoint_file}")
        else:
            print("No checkpoint found. Starting training from scratch.")

    # -----------------------------
    # 8. Training Loop with Checkpointing
    # -----------------------------
    for epoch in range(start_epoch, NUM_EPOCHS):
        TEACHER_FORCING_RATIO = max(0.25, 0.9 - epoch * 0.02)
        model.train()
        epoch_loss = 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            for batch_X, batch_y in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                predictions = model(batch_X)  # , target=batch_y, target_length=FORECAST_SEQ_LEN, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
                loss = criterion(predictions, batch_y)
                loss.backward()

                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)
                tepoch.set_postfix(loss=loss.item())
        epoch_loss /= len(train_loader.dataset)
        wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss})

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_truths = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_X.size(0)

                # Accumulate predictions and ground truths.
                all_val_preds.append(predictions.cpu().numpy())
                all_val_truths.append(batch_y.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        wandb.log({"epoch": epoch + 1, "val_loss": val_loss})

        # Concatenate all predictions and truths.

        val_preds = np.concatenate(all_val_preds, axis=0)
        val_truths = np.concatenate(all_val_truths, axis=0)

        # Compute additional metrics.
        evaluator = ForecastEvaluator(val_truths.flatten(), val_preds.flatten())
        eval_metrics = evaluator.evaluate_all()

        # Log the computed metrics.
        wandb.log({
            "R2": eval_metrics["R2"],
            "NMAE": eval_metrics["NMAE"],
            "NRMSE": eval_metrics["NRMSE"],
            "Skill Score": eval_metrics["Skill Score"]
        })

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Update the learning rate based on the validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr})
        print(f"Updated learning rate: {current_lr:.6f}")

        # Save checkpoint at the end of every epoch
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_loss
            }

            checkpoint_filename = f'checkpoint_{epoch + 1}.pt'
            torch.save(checkpoint, PATH_CHECKPOINT / checkpoint_filename)
            print(f"Checkpoint saved at epoch {epoch + 1} as {checkpoint_filename}")

    wandb.finish()


if __name__ == "__main__":
    main()
