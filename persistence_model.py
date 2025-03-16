import pandas as pd
import numpy as np
from tqdm import tqdm
from ForecastEvaluator import ForecastEvaluator  # your utility class for metrics
from model import data_loader, DEVICE

# Loop over the test loader and process one batch for evaluation.
for batch_X, batch_y in tqdm(data_loader.test_loader, desc="Evaluating"):
    # Move data to the proper device
    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

    # For this example, we use the first sample in the batch for evaluation.
    # Extract actual forecast target values from batch_y.
    # Assumed shape: (forecast_horizon, num_features)
    actual_data_tensor = batch_y[0]
    actual_data = pd.DataFrame(actual_data_tensor.cpu().numpy(), columns=['DNI', 'DHI'])

    # Get the forecast horizon (e.g., 60 minutes)
    forecast_horizon = actual_data.shape[0]

    # Create a dummy datetime index (adjust as needed if your data contains timestamps)
    date_index = pd.date_range(start="2025-03-16 12:00", periods=forecast_horizon, freq="1min")
    actual_data.index = date_index

    # Extract the last observation from the historical input.
    # Assumed shape of batch_X: (batch_size, history_length, num_features)
    last_observation_tensor = batch_X[0, -1]  # take the last time step of the first sample
    last_observation = {
        'DNI': last_observation_tensor[0].item(),
        'DHI': last_observation_tensor[1].item()
    }

    # Build the persistence forecast: repeat the last observation for each future time step.
    persistence_forecast = pd.DataFrame({
        'DNI': [last_observation['DNI']] * forecast_horizon,
        'DHI': [last_observation['DHI']] * forecast_horizon
    }, index=date_index)

    # Evaluate metrics for DNI and DHI separately using the ForecastEvaluator class.
    results = {}
    for target in ['DNI', 'DHI']:
        evaluator = ForecastEvaluator(actual_data[target], persistence_forecast[target])
        results[target] = evaluator.evaluate_all()

    # Display the results
    print("Evaluation Metrics:")
    for target, metrics in results.items():
        print(f"\n{target}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # For this example, process only the first batch.
    break
