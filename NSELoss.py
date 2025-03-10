import torch
import torch.nn as nn


class NashSutcliffeEfficiencyLoss(nn.Module):
    """
    Custom loss function for Nash–Sutcliffe Efficiency (NSE).

    NSE is defined as:
      NSE = 1 - (sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2))

    Since higher NSE values are better (with a maximum of 1), we define the loss as:
      loss = 1 - NSE

    This loss function can be used in PyTorch models to encourage predictions
    that maximize the NSE.
    """

    def __init__(self, epsilon: float = 1e-6):
        super(NashSutcliffeEfficiencyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate the numerator: sum of squared differences between targets and predictions.
        numerator = torch.sum((targets - predictions) ** 2)
        # Calculate the denominator: sum of squared differences from the mean of targets.
        denominator = torch.sum((targets - torch.mean(targets)) ** 2)
        # Add a small epsilon to the denominator to avoid division by zero.
        nse = 1 - numerator / (denominator + self.epsilon)
        # Since we want to maximize NSE, we define the loss as (1 - NSE).
        loss = 1 - nse
        return loss