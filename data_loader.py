from enum import Enum

import numpy as np
import pandas as pd

from _config import PKL_PROCESSED_STEP1_DTU_SOLAR_STATION, PATH_TO_CONFIG
from _utils import create_sequences, scale_dataframe
from my_config import load_config, MyConfig

class SPLIT(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3



class MyDataLoader:
    config:MyConfig
    df: pd.DataFrame
    mask: pd.DataFrame
    def __init__(self, config: MyConfig):
        self.config: MyConfig = config
        self.df: pd.DataFrame = None
        self.mask: pd.DataFrame = None


    def load_data(self):
        df = pd.read_pickle(PKL_PROCESSED_STEP1_DTU_SOLAR_STATION)
        df = df[sorted(df.columns)]
        self.df = df.copy()

    def reindex_full_range(self) -> None:
        full_range = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq="1min")
        self.df = self.df.reindex(full_range)
        self.df.sort_index(inplace=True)

    def lag_features(self):
        for feature in self.config.features:
            for lag in feature.lag:
                self.df[lag.name] = self.df[feature.name].shift(lag.value)

    def prepare_df(self, drop_solar_altitude_below_0: bool = True, drop_nan: bool = True) -> None:
        if drop_solar_altitude_below_0:
            # drop entire row if solar altitude is below 0
            mask = self.df['solar_altitude'] < 0
            self.df = self.df[~mask]

        if drop_nan:
            self.df.dropna(inplace=True)

    def get_df(self) -> pd.DataFrame:
        # All columns in the config should be present in the DataFrame
        if not set(self.config.get_df_names_from_config()).issubset(self.df.columns):
            raise Exception("Warning: DataFrame columns do not match config names. Please run lag_features() first.")

        df = self.df[self.config.get_df_names_from_config()].copy()
        return df

    def get_split(self, split: SPLIT) -> pd.DataFrame:
        df = self.get_df().copy()
        if split == SPLIT.TRAIN:
            return df.loc[self.config.train_val_test_split.train.start:self.config.train_val_test_split.train.end]
        elif split == SPLIT.VAL:
            return df.loc[self.config.train_val_test_split.val.start:self.config.train_val_test_split.val.end]
        elif split == SPLIT.TEST:
            return df.loc[self.config.train_val_test_split.test.start:self.config.train_val_test_split.test.end]
        else:
            raise ValueError("Invalid split type. Use SPLIT.TRAIN, SPLIT.VAL, or SPLIT.TEST.")

    def get_X_y(self, split: SPLIT, input_seq_len=60, rolling=True, verbose=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        df_train = self.get_split(split)
        X = df_train.drop(columns=self.config.TARGETS)
        y = df_train[self.config.TARGETS]

        if verbose: print(f'Before creating sequences: X shape: {X.shape}, y shape: {y.shape}')
        X, y, ts = create_sequences(X.to_numpy(), y.to_numpy(), df_train.index.to_numpy(), input_seq_len, rolling=rolling)
        if verbose: print(f'After creating sequences: X shape: {X.shape}, y shape: {y.shape}')
        return X, y, ts

    def get_feature_names(self) -> list[str]:
        return self.config.get_df_names_from_config(include_targets=False)

    def get_target_names(self) -> list[str]:
        return self.config.TARGETS

    def scale(self, filename:str, method: str = 'minmax') -> None:
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Please call load_data() first.")

        self.df = scale_dataframe(filename, self.df, method=method, columns=self.get_feature_names())


if __name__ == '__main__':
    my_config: MyConfig = load_config(PATH_TO_CONFIG)
    data_loader = MyDataLoader(my_config)
    data_loader.load_data()
    print(data_loader.df.info())
    data_loader.reindex_full_range()
    print(data_loader.df.info())
    data_loader.lag_features()
    print(data_loader.df.info())
    print(data_loader.get_df().info())
    data_loader.prepare_df(drop_solar_altitude_below_0=True, drop_nan=True)
    print(data_loader.df.info())
    print(data_loader.get_df().info())
    print(data_loader.get_split(SPLIT.TRAIN).info())
    print(data_loader.get_split(SPLIT.VAL).info())
    print(data_loader.get_split(SPLIT.TEST).info())
