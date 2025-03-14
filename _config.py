from pathlib import Path

PROJECT_ROOT = Path(r'D:\Jetbrains\Python\Projects\solar_irradiance_nowcasting')
PATH_RAW_DATA = PROJECT_ROOT / 'data' / 'raw'
PATH_PROCESSED_DATA = PROJECT_ROOT / 'data' / 'processed'

PATH_RAW_DTU_SOLAR_STATION = PATH_RAW_DATA / 'dtu_solar_station'
FILE_PROCESSED_DTU_SOLAR_STATION = PATH_PROCESSED_DATA / 'dtu_solar_station' / "preprocessed_data.pkl"
GAP_THRESHOLD = 10
"""
This gap threshold is used to skip sequences where the time difference between consecutive timestamps is larger than this value.

This allows a few missing values within the threshold to be kept as a sequence, but if the gap is too large, the sequence is skipped.
"""

for path in [PATH_OUTPUT, PATH_CHECKPOINT, PATH_RAW_DATA, PATH_PROCESSED_DATA, PATH_RAW_DTU_SOLAR_STATION]:
    path.mkdir(parents=True, exist_ok=True)