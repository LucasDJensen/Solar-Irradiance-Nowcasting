import pandas as pd
from astral import LocationInfo, Observer
from astral.sun import elevation

from _config import PATH_RAW_DTU_SOLAR_STATION, FILE_PROCESSED_DTU_SOLAR_STATION

PATH_RAW_DTU_SOLAR_STATION.mkdir(parents=True, exist_ok=True)
FILE_PROCESSED_DTU_SOLAR_STATION.parent.mkdir(parents=True, exist_ok=True)

# Get all CSV files in the path folder
csv_files = list(PATH_RAW_DTU_SOLAR_STATION.glob("*.csv"))

# Read each CSV file and combine them into a single DataFrame
dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

df.set_index('Time(utc)', inplace=True)
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

# Define location information for DTU Lyngby (Building 119)
location = LocationInfo(
    name="DTU Lyngby (Building 119)",
    region="Denmark",
    latitude=55.79064,
    longitude=12.52505,
)
observer = Observer(location.latitude, location.longitude, 50)  # 50 meters above sea level


# Define a function to compute solar altitude at a given datetime
def compute_solar_altitude(dt):
    return elevation(observer, dt)


# Apply the function to each datetime in the index and add a new column
df['solar_altitude'] = df.index.to_series().apply(compute_solar_altitude)

# Save the combined DataFrame to a pickle file
df.to_pickle(FILE_PROCESSED_DTU_SOLAR_STATION)
