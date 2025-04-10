import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ForecastEvaluator import ForecastEvaluator  # Utility class for evaluation metrics
from _config import PATH_CHECKPOINT, DATA_ROOT
from models import SimpleLSTM


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, device):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)  # [samples, seq_len, features]
        self.y = torch.tensor(y, dtype=torch.float32).to(device)  # [samples, forecast_seq_len, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(device, batch_size):
    train_data = np.load(DATA_ROOT / 'train.npz')
    X_train, y_train = train_data['X'], train_data['y']
    print(f'X shape: {X_train.shape}, y shape: {y_train.shape}')
    train_dataset = TimeSeriesDataset(X_train, y_train, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    del train_data, X_train, y_train  # Free up memory
    # Load the validation data
    val_data = np.load(DATA_ROOT / 'val.npz')
    X_val, y_val = val_data['X'], val_data['y']
    print(f'X_val shape: {X_val.shape}, y_val shape: {y_val.shape}')
    val_dataset = TimeSeriesDataset(X_val, y_val, device=device)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    del val_data, X_val, y_val  # Free up memory
    return train_dataset, train_loader, val_dataset, val_loader


def load_checkpoint(device, model, optimizer):
    # Find all checkpoint files in the directory following the pattern checkpoint_{epoch}.pt
    checkpoint_files = [f for f in os.listdir(PATH_CHECKPOINT) if f.startswith('checkpoint_') and f.endswith('.pt')]
    if checkpoint_files:
        # Sort files by epoch number in descending order and choose the latest checkpoint.
        checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
        latest_checkpoint_file = checkpoint_files[0]

        checkpoint = torch.load(PATH_CHECKPOINT / latest_checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        print(f"Resuming training from epoch {start_epoch} using checkpoint {latest_checkpoint_file}")
    else:
        print("No checkpoint found. Starting training from scratch.")
        start_epoch = 1
    return start_epoch


def train_model(criterion, device, epoch, model, optimizer, train_loader, clip_grad_norm):
    model.train()
    epoch_loss = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_X, batch_y in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)
            tepoch.set_postfix(loss=loss.item())
    epoch_loss /= len(train_loader)
    wandb.log({"epoch": epoch, "train_loss": epoch_loss, "lr": optimizer.param_groups[0]["lr"]})
    return epoch_loss


def validate_model(criterion, device, epoch, model, val_loader):
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
    val_loss /= len(val_loader)
    wandb.log({"epoch": epoch, "val_loss": val_loss})
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
        "Skill Score": eval_metrics["Skill Score"],
        "RMSE": eval_metrics["RMSE"],
        "MAE": eval_metrics["MAE"],
        "MBE": eval_metrics["MBE"]
    })

    # Log example prediction
    if epoch % 1 == 0:  # or every epoch, your choice
        idx = np.random.randint(0, val_preds.shape[0])
        plt.figure(figsize=(10, 4))
        plt.plot(val_truths[idx], label='Truth')
        plt.plot(val_preds[idx], label='Prediction')
        plt.legend()
        plt.title(f'Validation Prediction Example - Epoch {epoch}')
        wandb.log({"val_example_plot": wandb.Image(plt)})
        plt.close()
    return val_loss


def save_checkpoint(epoch, epoch_loss, model, optimizer, val_loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_loss,
        'val_loss': val_loss
    }
    checkpoint_filename = f'checkpoint_{epoch}.pt'
    torch.save(checkpoint, PATH_CHECKPOINT / checkpoint_filename)
    print(f"Checkpoint saved at epoch {epoch} as {checkpoint_filename}")


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_SIZE = 8  # Update if you're using multivariate input
    OUTPUT_SIZE = 2
    EPOCHS = 200
    PROJECT_NAME = "solar-irradiance-nowcasting"

    TARGETS = ['DNI', 'DHI']
    INPUT_SEQ_LEN = 60  # Past 60 minutes as input

    NUM_LSTM_LAYERS = 1
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = len(TARGETS)  # predicting two features (DNI, DHI)

    BATCH_SIZE = 128

    RESUME = False

    LEARNING_RATE = 1e-3
    DROPOUT = 0.5
    clip_grad_norm = 1

    train_dataset, train_loader, val_dataset, val_loader = load_dataset(DEVICE, BATCH_SIZE)

    input_size = train_dataset.X.shape[2]  # number of features per time step
    model = SimpleLSTM(input_size, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LSTM_LAYERS, dropout=DROPOUT).to(DEVICE)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    wandb.init(project=PROJECT_NAME, entity="s210025-dtu",
               config={"Dataset": "DTU Solar Station",
                       "Model": "SimpleLSTM",
                       "Input Sequence Length": INPUT_SEQ_LEN,
                       "Hidden Size": HIDDEN_SIZE,
                       "LSTM Layers": NUM_LSTM_LAYERS,
                       "Batch Size": BATCH_SIZE,
                       "Learning Rate": LEARNING_RATE,
                       "Targets": TARGETS,
                       "Output Size": OUTPUT_SIZE,
                       "Loss": criterion,
                       "Optimizer": optimizer,
                       "Dropout": DROPOUT,
                       })

    wandb.watch(model, log="all")

    start_epoch = 1

    if RESUME:
        start_epoch = load_checkpoint(DEVICE, model, optimizer)

    for epoch in range(start_epoch, EPOCHS + 1):

        train_loss = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            clip_grad_norm=clip_grad_norm  # Pass it in if your train_loop supports it
        )
        val_loss = validate_model(
            criterion=criterion,
            device=DEVICE,
            epoch=epoch,
            model=model,
            val_loader=val_loader
        )
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if epoch % 10 == 0:
            save_checkpoint(epoch, train_loss, model, optimizer, val_loss)

    wandb.finish()


if __name__ == "__main__":
    main()
