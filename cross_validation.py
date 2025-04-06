def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

# Grid search parameters
param_grid = {
    'hidden_size': [32, 64],
    'num_layers': [1, 2],
    'lr': [0.001, 0.0005],
    'batch_size': [32],
    'epochs': [30]
}

# Time series split
tscv = TimeSeriesSplit(n_splits=5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_score = float('inf')
best_params = None

for params in ParameterGrid(param_grid):
    val_losses = []

    for train_idx, val_idx in tscv.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

        model = LSTMModel(1, params['hidden_size'], params['num_layers']).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

        for epoch in range(params['epochs']):
            train_model(model, train_loader, criterion, optimizer, device)

        val_loss = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

    avg_val_loss = np.mean(val_losses)
    print(f"Params: {params}, Val Loss: {avg_val_loss:.6f}")

    if avg_val_loss < best_score:
        best_score = avg_val_loss
        best_params = params
