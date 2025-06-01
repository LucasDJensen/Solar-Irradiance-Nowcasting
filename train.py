import datetime
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
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
PATH_CHECKPOINT = PATH_CHECKPOINT / time_str

my_config: MyConfig = load_config(PATH_TO_CONFIG)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100

TARGETS = my_config.TARGETS
FEATURES = my_config.FEATURES
INPUT_SEQ_LEN = 60  # Past x minutes as input

NUM_LSTM_LAYERS = 3
HIDDEN_SIZE = 32
OUTPUT_SIZE = len(TARGETS)
INPUT_SIZE = len(FEATURES)

BATCH_SIZE = 512

LEARNING_RATE = 1e-3
DROPOUT = 0.15
CLIP_GRAD_NORM = 0.835

RESUME = False

model = SimpleLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LSTM_LAYERS, dropout=DROPOUT).to(DEVICE)


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
    # criterion = nn.SmoothL1Loss()

    criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
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

    wandb.init(project="solar-irradiance-nowcasting", entity="s210025-dtu",
               config={"Dataset": "DTU Solar Station",
                       "Model": "SimpleLSTM",
                       "Input Sequence Length": INPUT_SEQ_LEN,
                       "Hidden Size": HIDDEN_SIZE,
                       "LSTM Layers": NUM_LSTM_LAYERS,
                       "Batch Size": BATCH_SIZE,
                       "Learning Rate": LEARNING_RATE,
                       "Gradient clipping": CLIP_GRAD_NORM,
                       "Targets": TARGETS,
                       "Output Size": OUTPUT_SIZE,
                       "Loss": criterion,
                       "Optimizer": optimizer,
                       "Dropout": DROPOUT,
                       })

    wandb.watch(model, log="all")

    start_epoch = 1

    train_dataset, train_loader, val_dataset, val_loader = load_dataset(my_config, DEVICE, BATCH_SIZE)

    for epoch in range(start_epoch, EPOCHS + 1):

        train_loss = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            clip_grad_norm=CLIP_GRAD_NORM  # Pass it in if your train_loop supports it
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

        if epoch % 5 == 0:
            save_checkpoint(epoch, train_loss, model, optimizer, val_loss)

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


if __name__ == "__main__":
    PATH_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    shutil.copy(PATH_TO_CONFIG, PATH_CHECKPOINT / 'config.json')  # copy config file to checkpoint directory
    shutil.copy(__file__, PATH_CHECKPOINT / 'train.py')  # copy training script to checkpoint directory
    shutil.copy(os.path.join(os.path.dirname(__file__), 'models.py'), PATH_CHECKPOINT / 'models.py')  # copy models.py to checkpoint directory
    main()
