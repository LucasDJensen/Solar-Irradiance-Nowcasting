import datetime
import json
import os
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import optuna
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from ForecastEvaluator import ForecastEvaluator  # Utility class for evaluation metrics
from TimeSeriesDataset import TimeSeriesDataset
from _config import PATH_CHECKPOINT
potential_config = Path('./config.json')
if potential_config.exists():
    PATH_TO_CONFIG = potential_config
else:
    from _config import PATH_TO_CONFIG
from _utils import load_scaler_and_transform_df, scale_dataframe
from data_loader import MyDataLoader, SPLIT
from models import *
from my_config import load_config, MyConfig


def load_dataset(my_config: MyConfig, path_checkpoint):
    data_loader = MyDataLoader(my_config)
    data_loader.load_data()
    data_loader.load_ecmwf_data()
    data_loader.reindex_full_range()
    data_loader.lag_features()
    data_loader.prepare_df(drop_solar_altitude_below_0=True, drop_nan=True)

    method = 'minmax'  # or 'standard', 'maxabs', 'robust', 'normalizer'
    scalar_file = path_checkpoint / f'{method}.pkl'

    # 1) SCALE train split and convert to NumPy
    df_train = scale_dataframe(
        filename=scalar_file,
        df=data_loader.get_split(SPLIT.TRAIN),
        method=method,
        columns=data_loader.get_feature_names() + data_loader.get_target_names()
    )
    X_train_df = df_train[data_loader.get_feature_names()]  # pandas.DataFrame
    y_train_df = df_train[data_loader.get_target_names()]  # pandas.DataFrame
    ts_train = df_train.index.to_numpy()  # Index → np.datetime64[...]

    X_train_full = X_train_df.to_numpy()  # dtype likely float64 → we’ll cast inside Dataset
    y_train_full = y_train_df.to_numpy()

    # 2) BUILD TimeSeriesDataset for TRAIN
    train_dataset = TimeSeriesDataset(
        X_full=X_train_full,
        y_full=y_train_full,
        ts_full=ts_train,
        input_seq_len=my_config.INPUT_SEQ_LEN,
        gap_threshold_minutes=my_config.GAP_THRESHOLD
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=my_config.BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        pin_memory=True
    )
    del y_train_full, X_train_full, ts_train, df_train

    # 3) SCALE val split using existing scaler, convert to NumPy
    df_val = load_scaler_and_transform_df(scalar_file, data_loader.get_split(SPLIT.VAL))
    X_val_df = df_val[data_loader.get_feature_names()]
    y_val_df = df_val[data_loader.get_target_names()]
    ts_val = df_val.index.to_numpy()

    X_val_full = X_val_df.to_numpy()
    y_val_full = y_val_df.to_numpy()

    val_dataset = TimeSeriesDataset(
        X_full=X_val_full,
        y_full=y_val_full,
        ts_full=ts_val,
        input_seq_len=my_config.INPUT_SEQ_LEN,
        gap_threshold_minutes=my_config.GAP_THRESHOLD
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=my_config.BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        pin_memory=True
    )
    del y_val_full, X_val_full, ts_val, df_val
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    return train_dataset, train_loader, val_dataset, val_loader


def train_model(criterion, device, epoch, model, optimizer, train_loader, clip_grad_norm):
    model.train()
    epoch_loss = 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_X, batch_y, _ in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)
            tepoch.set_postfix(loss=loss.item())
    epoch_loss /= len(train_loader.dataset)
    wandb.log({"epoch": epoch, "train_loss": epoch_loss, "lr": optimizer.param_groups[0]["lr"]})
    return epoch_loss


def validate_model(criterion, device, epoch, model, val_loader):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_truths = []
    with torch.no_grad():
        for batch_X, batch_y, _ in val_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            val_loss += loss.item() * batch_X.size(0)

            all_preds.append(preds.cpu().numpy())
            all_truths.append(batch_y.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    wandb.log({"epoch": epoch, "val_loss": val_loss})

    val_preds = np.concatenate(all_preds, axis=0)
    val_truths = np.concatenate(all_truths, axis=0)
    evaluator = ForecastEvaluator(val_truths.flatten(), val_preds.flatten())
    eval_metrics = evaluator.evaluate_all()
    wandb.log(eval_metrics)

    # Example plot (unchanged)
    idx = np.random.randint(0, val_preds.shape[0])
    plt.figure(figsize=(10, 4))
    plt.plot(val_truths[idx], label="Truth")
    plt.plot(val_preds[idx], label="Prediction")
    plt.legend()
    plt.title(f"Validation Example (Epoch {epoch})")
    wandb.log({"val_example_plot": wandb.Image(plt)})
    plt.close()
    return val_loss


time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

my_config: MyConfig = load_config(PATH_TO_CONFIG)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Objective function for Optuna
def objective(trial):
    # Sample hyperparameters
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512])
    clip_grad_norm = trial.suggest_float("clip_grad_norm", 0.6, 0.8)

    train_dataset, train_loader, val_dataset, val_loader = load_dataset(my_config, PATH_CHECKPOINT)

    model_state = {
        'input_size': len(my_config.get_df_names_from_config(include_targets=False)),
        'hidden_size': hidden_size,
        'output_size': len(my_config.get_df_target_names()),
        'num_layers': num_layers,
        'dropout': dropout
    }

    model = SimpleLSTM(input_size=model_state['input_size'],
                       hidden_size=model_state['hidden_size'],
                       output_size=model_state['output_size'],
                       num_layers=model_state['num_layers'],
                       dropout=model_state['dropout']).to(DEVICE)


    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=1, min_lr=1e-7
    )

    wandb.init(project="solar-nowcasting-optuna",
               config={"Dataset": "DTU Solar Station",
                       "Model": "SimpleLSTM",
                       "Input Sequence Length": my_config.INPUT_SEQ_LEN,
                       "Hidden Size": hidden_size,
                       "LSTM Layers": num_layers,
                       "Batch Size": batch_size,
                       "Learning Rate": lr,
                       "Gradient clipping": clip_grad_norm,
                       "Targets": my_config.get_df_target_names(),
                       "Features": my_config.get_df_names_from_config(include_targets=False),
                       "Loss": criterion,
                       "Optimizer": optimizer,
                       "Dropout": dropout,
                       },
               reinit=True
               )

    wandb.watch(model, log="all")
    start_epoch = 1

    best_val = float('inf')
    no_improve = 0

    for epoch in range(start_epoch, my_config.EPOCHS + 1):

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

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})

        print(f"Epoch {epoch}/{my_config.EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # simple early stopping
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= my_config.EARLY_STOPPING_PATIENCE:
                print(f"No improvement for {my_config.EARLY_STOPPING_PATIENCE} epochs, stopping early.")
                break
            print(f"Current LR: {scheduler.optimizer.param_groups[0]['lr']:.2e}")


    wandb.finish()
    return best_val


# Run the sweep
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best trial:")
    print(study.best_trial.params)
