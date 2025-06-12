import datetime
import json
import os
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from ForecastEvaluator import ForecastEvaluator  # Utility class for evaluation metrics
from TimeSeriesDataset import TimeSeriesDatasetGPU
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

def load_dataset(my_config: MyConfig, path_checkpoint, device: torch.device):
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

    X_train_full = X_train_df.to_numpy().astype(np.float32)
    y_train_full = y_train_df.to_numpy().astype(np.float32)

    # 2) BUILD TimeSeriesDatasetGPU for TRAIN (pass device here!)
    train_dataset = TimeSeriesDatasetGPU(
        X_full=X_train_full,
        y_full=y_train_full,
        ts_full=ts_train,
        input_seq_len=my_config.INPUT_SEQ_LEN,
        gap_threshold_minutes=my_config.GAP_THRESHOLD,
        device=device
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=my_config.BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        pin_memory=False,  # pin_memory can be left False—batches are already on GPU
        num_workers=0  # <=== Set to 0 to avoid duplicating GPU tensors across workers
    )
    del y_train_full, X_train_full, ts_train, df_train

    # 3) SCALE val split and build val dataset similarly:
    df_val = load_scaler_and_transform_df(scalar_file, data_loader.get_split(SPLIT.VAL))
    X_val_df = df_val[data_loader.get_feature_names()]
    y_val_df = df_val[data_loader.get_target_names()]
    ts_val = df_val.index.to_numpy()

    X_val_full = X_val_df.to_numpy().astype(np.float32)
    y_val_full = y_val_df.to_numpy().astype(np.float32)

    val_dataset = TimeSeriesDatasetGPU(
        X_full=X_val_full,
        y_full=y_val_full,
        ts_full=ts_val,
        input_seq_len=my_config.INPUT_SEQ_LEN,
        gap_threshold_minutes=my_config.GAP_THRESHOLD,
        device=device
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=my_config.BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        pin_memory=False,
        num_workers=0
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
            optimizer.zero_grad()
            preds = model(batch_X)  # both X and model on GPU
            loss = criterion(preds, batch_y)  # no device copy needed
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
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            val_loss += loss.item() * batch_X.size(0)

            all_preds.append(preds.cpu().numpy())  # move to CPU only for logging
            all_truths.append(batch_y.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    wandb.log({"epoch": epoch, "val_loss": val_loss})

    val_preds = np.concatenate(all_preds, axis=0)
    val_truths = np.concatenate(all_truths, axis=0)
    evaluator = ForecastEvaluator()
    eval_metrics = evaluator.evaluate_all(val_truths.flatten(), val_preds.flatten())
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


if __name__ == "__main__":
    my_config: MyConfig = load_config(PATH_TO_CONFIG)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S_{0}_{1}_{2}".format(my_config.name, my_config.NUM_LSTM_LAYERS, my_config.HIDDEN_SIZE))
    PATH_CHECKPOINT = PATH_CHECKPOINT / time_str

    PATH_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    shutil.copy(PATH_TO_CONFIG, PATH_CHECKPOINT / 'config.json')  # copy config file to checkpoint directory
    shutil.copy(__file__, PATH_CHECKPOINT / 'train.py')  # copy training script to checkpoint directory
    shutil.copy(os.path.join(os.path.dirname(__file__), 'models.py'), PATH_CHECKPOINT / 'models.py')  # copy models.py to checkpoint directory


    train_dataset, train_loader, val_dataset, val_loader = load_dataset(my_config, PATH_CHECKPOINT, DEVICE)
    model_state = {
        'input_size': len(my_config.get_df_names_from_config(include_targets=False)),
        'hidden_size': my_config.HIDDEN_SIZE,
        'output_size': len(my_config.get_df_target_names()),
        'num_layers': my_config.NUM_LSTM_LAYERS,
        'dropout': my_config.DROPOUT
    }
    # save model state to file
    with open(PATH_CHECKPOINT / 'model_architecture.json', 'w') as f:
        json.dump(model_state, f)
    
    model = SimpleLSTM(input_size=model_state['input_size'],
                       hidden_size=model_state['hidden_size'],
                       output_size= model_state['output_size'],
                       num_layers=model_state['num_layers'],
                       dropout=model_state['dropout']).to(DEVICE)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=my_config.LEARNING_RATE, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-8
    )

    wandb.init(project="solar-irradiance-nowcasting", entity="s210025-dtu",
               config={"Dataset": "DTU Solar Station",
                       "Model": "SimpleLSTM",
                       "Config Name": my_config.name,
                       "Input Sequence Length": my_config.INPUT_SEQ_LEN,
                       "Hidden Size": my_config.HIDDEN_SIZE,
                       "LSTM Layers": my_config.NUM_LSTM_LAYERS,
                       "Batch Size": my_config.BATCH_SIZE,
                       "Learning Rate": my_config.LEARNING_RATE,
                       "Gradient clipping": my_config.CLIP_GRAD_NORM,
                       "Targets": my_config.get_df_target_names(),
                       "Features": my_config.get_df_names_from_config(include_targets=False),
                       "Loss": criterion,
                       "Optimizer": optimizer,
                       "Dropout": my_config.DROPOUT,
                       })

    wandb.watch(model, log="all")

    best_val = float('inf')
    no_improve = 0

    for epoch in range(1, my_config.EPOCHS + 1):

        train_loss = train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
            clip_grad_norm=my_config.CLIP_GRAD_NORM  # Pass it in if your train_loop supports it
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

        if epoch % 5 == 0:
            save_checkpoint(epoch, train_loss, model, optimizer, val_loss)

        # simple early stopping
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= my_config.EARLY_STOPPING_PATIENCE:
                print(f"No improvement for {my_config.EARLY_STOPPING_PATIENCE} epochs, stopping early.")
                save_checkpoint(epoch, train_loss, model, optimizer, val_loss)
                break
            print(f"Current LR: {scheduler.optimizer.param_groups[0]['lr']:.2e}")

    wandb.finish()
