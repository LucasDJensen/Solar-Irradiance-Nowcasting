import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class ForecastEvaluator:
    def __init__(self, y_true, y_pred):
        """
        Initializes the evaluator with true and predicted values.

        Parameters:
            y_true (array-like): The actual observed values.
            y_pred (array-like): The forecasted values.
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    @staticmethod
    def compute_r2(y_true, y_pred):
        """
        Compute the coefficient of determination (R²).
        """
        return r2_score(y_true, y_pred)

    @staticmethod
    def compute_nmae(y_true, y_pred):
        """
        Compute the normalized mean absolute error (NMAE).
        The error is normalized by the range of the true values.
        """
        mae = mean_absolute_error(y_true, y_pred)
        y_range = y_true.max() - y_true.min()
        return mae / y_range if y_range != 0 else np.nan

    @staticmethod
    def compute_nrmse(y_true, y_pred):
        """
        Compute the normalized root mean squared error (NRMSE).
        The error is normalized by the range of the true values.
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        y_range = y_true.max() - y_true.min()
        return rmse / y_range if y_range != 0 else np.nan

    @staticmethod
    def compute_skill_score(y_true, y_pred):
        """
        Compute the forecast skill score (SS) relative to a climatology forecast.
        The climatology is defined as the mean of the true values.
        """
        mse_model = mean_squared_error(y_true, y_pred)
        # Climatology forecast: using the mean of the observed values.
        climatology = np.full_like(y_true, fill_value=y_true.mean())
        mse_climatology = mean_squared_error(y_true, climatology)
        return 1 - (mse_model / mse_climatology) if mse_climatology != 0 else np.nan

    def evaluate_all(self):
        """
        Compute and return all evaluation metrics as a dictionary.
        """
        metrics = {
            'R2': self.compute_r2(self.y_true, self.y_pred),
            'NMAE': self.compute_nmae(self.y_true, self.y_pred),
            'NRMSE': self.compute_nrmse(self.y_true, self.y_pred),
            'Skill Score': self.compute_skill_score(self.y_true, self.y_pred)
        }
        return metrics