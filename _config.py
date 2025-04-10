from pathlib import Path

PROJECT_ROOT = Path(r'/home/s210025/Solar-Irradiance-Nowcasting')
PATH_OUTPUT = PROJECT_ROOT / 'output'
PATH_CHECKPOINT = PROJECT_ROOT / 'checkpoint'
FILE_SCALARS = PATH_OUTPUT / 'scalers.pkl'
DATA_ROOT = Path(r'/dtu-compute/s210025')
PATH_RAW_DATA = DATA_ROOT / 'data' / 'raw'
PATH_PROCESSED_DATA = DATA_ROOT / 'data' / 'processed'

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