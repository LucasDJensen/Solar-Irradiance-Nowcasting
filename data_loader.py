import pandas as pd

from _config import PKL_PROCESSED_STEP1_DTU_SOLAR_STATION, PATH_TO_CONFIG
from my_config import load_config, MyConfig


class MyDataLoader:
    def __init__(self, config: MyConfig):
        self.config: MyConfig = config
        self.df: pd.DataFrame = None

    def load_data(self):
        df = pd.read_pickle(PKL_PROCESSED_STEP1_DTU_SOLAR_STATION)
        df = df[sorted(df.columns)]
        self.df = df.copy()

    def lag_features(self):
        for feature in self.config.features:
            for lag in feature.lag:
                self.df[lag.name] = self.df[feature.name].shift(lag.value)

    def prepare_df(self, drop_solar_altitude_below_0: bool = True, drop_nan: bool = True):
        if drop_solar_altitude_below_0:
            # drop entire row if solar altitude is below 0
            mask = self.df['solar_altitude'] < 0
            self.df = self.df[~mask]

        if drop_nan:
            # Drop rows with any NaN values in any column
            self.df = self.df.dropna()

    def get_df(self):
        df = self.df[self.config.get_df_names_from_config()].copy()
        return df


if __name__ == '__main__':
    my_config: MyConfig = load_config(PATH_TO_CONFIG)
    data_loader = MyDataLoader(my_config)
    data_loader.load_data()
    print(data_loader.df.info())
    data_loader.lag_features()
    print(data_loader.df.info())
    data_loader.prepare_df()
    print(data_loader.df.info())
