import os
from pathlib import Path
PATH_TO_CONFIG = Path(os.getenv("CONFIG", r'D:\Jetbrains\Python\Projects\solar_irradiance_nowcasting\configs\dni_only\dni_and_station.json'))

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", r'D:\Jetbrains\Python\Projects\solar_irradiance_nowcasting'))
PATH_OUTPUT = PROJECT_ROOT / 'output'
PATH_CHECKPOINT = PROJECT_ROOT / 'checkpoint'
FILE_SCALARS = PATH_OUTPUT / 'scalers.pkl'
DATA_ROOT = Path(os.getenv("DATA_ROOT", r'D:\Jetbrains\Python\Projects\solar_irradiance_nowcasting\data'))
PATH_RAW_DATA = DATA_ROOT / 'raw'
PATH_PROCESSED_DATA = DATA_ROOT / 'processed'

PATH_RAW_DTU_SOLAR_STATION = PATH_RAW_DATA / 'dtu_solar_station'
PKL_PROCESSED_STEP1_DTU_SOLAR_STATION = PATH_PROCESSED_DATA / 'dtu_solar_station' / "preprocessed_data_step1.pkl"
PKL_PROCESSED_STEP2_DTU_SOLAR_STATION = PATH_PROCESSED_DATA / 'dtu_solar_station' / "preprocessed_data_step2.pkl"
GAP_THRESHOLD = 1
"""
This gap threshold is used to skip sequences where the time difference between consecutive timestamps is larger than this value.

This allows a few missing values within the threshold to be kept as a sequence, but if the gap is too large, the sequence is skipped.
"""

for path in [PATH_OUTPUT, PATH_CHECKPOINT, PATH_RAW_DATA, PATH_PROCESSED_DATA, PATH_RAW_DTU_SOLAR_STATION]:
    path.mkdir(parents=True, exist_ok=True)