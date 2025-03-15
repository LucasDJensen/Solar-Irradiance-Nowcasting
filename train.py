import os

import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from _config import GAP_THRESHOLD, PATH_CHECKPOINT
from model import DEVICE, INPUT_SEQ_LEN, FORECAST_SEQ_LEN, HIDDEN_SIZE, NUM_LSTM_LAYERS, SPLIT, TARGETS, OUTPUT_SIZE, BATCH_SIZE, data_loader, input_size, model

# -----------------------------
# 1. Parameters and Hyperparameters
# -----------------------------
# Set to True if you want to resume training from a checkpoint.
RESUME = False

LEARNING_RATE = 1e-3
NUM_EPOCHS = 1
TEACHER_FORCING_RATIO = 0.7
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
                        "Teacher Forcing Ratio": TEACHER_FORCING_RATIO,
                        "Scheduler Patience": SCHEDULER_PATIENCE,
                        "Split": SPLIT,
                        "Gap Threshold": GAP_THRESHOLD,
                        "Targets": TARGETS,
                        "Output Size": OUTPUT_SIZE,
                        "Loss": criterion,
                        "Optimizer": optimizer,
                        "Scheduler": scheduler
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

        checkpoint = torch.load(PATH_CHECKPOINT / latest_checkpoint_file, map_location=DEVICE)
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

    with tqdm(data_loader.train_loader, unit="batch") as tepoch:
        for batch_X, batch_y in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(batch_X)  # , target=batch_y, target_length=FORECAST_SEQ_LEN, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
            loss = criterion(predictions, batch_y)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)
            tepoch.set_postfix(loss=loss.item())
    epoch_loss /= len(data_loader.train_dataset)
    wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss})

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in data_loader.val_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            predictions = model(batch_X)  # , target=None, target_length=FORECAST_SEQ_LEN, teacher_forcing_ratio=0.0)  # no teacher forcing during evaluation
            loss = criterion(predictions, batch_y)
            val_loss += loss.item() * batch_X.size(0)
    val_loss /= len(data_loader.val_dataset)
    wandb.log({"epoch": epoch + 1, "val_loss": val_loss})
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Update the learning rate based on the validation loss
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    wandb.log({"learning_rate": current_lr})
    print(f"Updated learning rate: {current_lr:.6f}")

    # Save checkpoint at the end of every epoch
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
