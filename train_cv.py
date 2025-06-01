import datetime
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import optuna
import wandb
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ForecastEvaluator import ForecastEvaluator  # Utility class for evaluation metrics
from _config import PATH_CHECKPOINT, PATH_TO_CONFIG
from _utils import load_scaler_and_transform_df, scale_dataframe
from data_loader import MyDataLoader, SPLIT
from models import *
from my_config import load_config, MyConfig

time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

my_config: MyConfig = load_config(PATH_TO_CONFIG)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100

TARGETS = my_config.TARGETS
FEATURES = my_config.FEATURES
INPUT_SEQ_LEN = 60  # Past x minutes as input

OUTPUT_SIZE = len(TARGETS)
INPUT_SIZE = len(FEATURES)


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, device):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)  # [samples, seq_len, features]
        self.y = torch.tensor(y, dtype=torch.float32).to(device)  # [samples, forecast_seq_len, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(my_config, device, batch_size):
    data_loader = MyDataLoader(my_config)
    data_loader.load_data()
    data_loader.reindex_full_range()
    data_loader.lag_features()
    data_loader.prepare_df(drop_solar_altitude_below_0=True, drop_nan=True)
    method = 'minmax'  # or 'standard', 'maxabs', 'robust', 'normalizer'
    scalar_file = PATH_CHECKPOINT / f'{method}.pkl'

    df_train = scale_dataframe(scalar_file, data_loader.get_split(SPLIT.TRAIN), method=method, columns=data_loader.get_feature_names() + data_loader.get_target_names())
    X_train, y_train, _ = data_loader.get_X_y(df_train, input_seq_len=INPUT_SEQ_LEN, rolling=True, verbose=True)
    train_dataset = TimeSeriesDataset(X_train, y_train, device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    df_val = load_scaler_and_transform_df(scalar_file, data_loader.get_split(SPLIT.VAL))
    X_val, y_val, ts = data_loader.get_X_y(df_val, input_seq_len=INPUT_SEQ_LEN, rolling=True, verbose=True)
    val_dataset = TimeSeriesDataset(X_val, y_val, device=device)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, train_loader, val_dataset, val_loader


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
    epoch_loss /= len(train_loader.dataset)
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
    val_loss /= len(val_loader.dataset)
    wandb.log({"epoch": epoch, "val_loss": val_loss})
    # Concatenate all predictions and truths.
    val_preds = np.concatenate(all_val_preds, axis=0)
    val_truths = np.concatenate(all_val_truths, axis=0)
    # Compute additional metrics.
    evaluator = ForecastEvaluator(val_truths.flatten(), val_preds.flatten())
    eval_metrics = evaluator.evaluate_all()
    # Log the computed metrics.
    wandb.log(eval_metrics)

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


# Objective function for Optuna
def objective(trial):
    # Sample hyperparameters
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])
    clip_grad_norm = trial.suggest_float("clip_grad_norm", 0.5, 1.0)

    # Initialize model
    model = SimpleLSTM(INPUT_SIZE, hidden_size, OUTPUT_SIZE, num_layers, dropout=dropout).to(DEVICE)
    # criterion = nn.SmoothL1Loss()

    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-3  # L2 regularization
    )
    # LR scheduler on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )
    best_val = float('inf')
    no_improve = 0
    EARLY_STOP_PATIENCE = 5

    wandb.init(project="solar-nowcasting-optuna",
               config={"Dataset": "DTU Solar Station",
                       "Model": "SimpleLSTM",
                       "Input Sequence Length": INPUT_SEQ_LEN,
                       "Hidden Size": hidden_size,
                       "LSTM Layers": num_layers,
                       "Batch Size": batch_size,
                       "Learning Rate": lr,
                       "Gradient clipping": clip_grad_norm,
                       "Targets": TARGETS,
                       "Output Size": OUTPUT_SIZE,
                       "Loss": criterion,
                       "Optimizer": optimizer,
                       "Dropout": dropout,
                       },
               reinit=True
               )
    start_epoch = 1

    train_dataset, train_loader, val_dataset, val_loader = load_dataset(my_config, DEVICE, batch_size)

    best_val_loss = float("inf")
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
        # step the scheduler
        scheduler.step(val_loss)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # simple early stopping
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"No improvement for {EARLY_STOP_PATIENCE} epochs, stopping early.")
                break
            print(f"Current LR: {scheduler.optimizer.param_groups[0]['lr']:.2e}")


    wandb.finish()
    return best_val_loss


# Run the sweep
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    print(study.best_trial.params)
