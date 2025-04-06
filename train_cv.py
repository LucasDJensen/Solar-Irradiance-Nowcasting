import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import optuna
from ForecastEvaluator import ForecastEvaluator  # Utility class for evaluation metrics
from _config import PATH_CHECKPOINT
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
    train_data = np.load('data/train.npz')
    X_train, y_train = train_data['X'], train_data['y']
    print(f'X shape: {X_train.shape}, y shape: {y_train.shape}')
    train_dataset = TimeSeriesDataset(X_train, y_train, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    del train_data, X_train, y_train  # Free up memory
    # Load the validation data
    val_data = np.load('data/val.npz')
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


# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 8  # Update if you're using multivariate input
OUTPUT_SIZE = 2
EPOCHS = 20  # You can increase for real sweep
PROJECT_NAME = "solar-nowcasting-optuna"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Objective function for Optuna
def objective(trial):
    # Sample hyperparameters
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    clip_grad_norm = trial.suggest_float("clip_grad_norm", 0.5, 5.0)

    # Initialize Weights & Biases
    wandb.init(
        project=PROJECT_NAME,
        config={
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "clip_grad_norm": clip_grad_norm,
            "epochs": EPOCHS
        },
        reinit=True
    )

    # Load data
    train_dataset, val_dataset, train_loader, val_loader = load_dataset(device, batch_size)

    # Initialize model
    model = SimpleLSTM(INPUT_SIZE, hidden_size, OUTPUT_SIZE, num_layers, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
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
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    wandb.finish()
    return best_val_loss


# Run the sweep
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    print(study.best_trial.params)
