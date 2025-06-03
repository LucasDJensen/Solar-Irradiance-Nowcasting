import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class ForecastEvaluator:

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
        climatology = np.full_like(y_true, fill_value=y_true.mean())
        mse_climatology = mean_squared_error(y_true, climatology)
        return 1 - (mse_model / mse_climatology) if mse_climatology != 0 else np.nan

    @staticmethod
    def compute_rmse(y_true, y_pred):
        """
        Compute the root mean squared error (RMSE).
        """
        mse = mean_squared_error(y_true, y_pred)
        return np.sqrt(mse)

    @staticmethod
    def compute_mae(y_true, y_pred):
        """
        Compute the mean absolute error (MAE).
        """
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def compute_mbe(y_true, y_pred):
        """
        Compute the mean bias error (MBE).
        """
        return np.mean(y_pred - y_true)

    @staticmethod
    def compute_mape(y_true, y_pred):
        """
        Compute the mean absolute percentage error (MAPE).
        """
        valid_indices = y_true != 0
        return np.mean(np.abs((y_true[valid_indices] - y_pred[valid_indices]) / y_true[valid_indices])) * 100


    def evaluate_all(self, y_true, y_pred):
        """
        Compute and return all evaluation metrics as a dictionary.
        """
        metrics = {
            'RMSE': self.compute_rmse(y_true, y_pred),
            'MAE': self.compute_mae(y_true, y_pred),
            'MAPE': self.compute_mape(y_true, y_pred),
            'R2': self.compute_r2(y_true, y_pred),
            'NMAE': self.compute_nmae(y_true, y_pred),
            'NRMSE': self.compute_nrmse(y_true, y_pred),
            'Skill Score': self.compute_skill_score(y_true, y_pred),
            'MBE': self.compute_mbe(y_true, y_pred),
        }
        return metrics