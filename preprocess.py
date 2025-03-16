import pandas as pd
from astral import LocationInfo, Observer
from astral.sun import elevation

from _config import PATH_RAW_DTU_SOLAR_STATION, FILE_PROCESSED_DTU_SOLAR_STATION, FILE_SCALARS

# Get all CSV files in the path folder
csv_files = list(PATH_RAW_DTU_SOLAR_STATION.glob("*.csv"))

# Read each CSV file and combine them into a single DataFrame
dfs = [pd.read_csv(f) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

# Remove the LWD column as it contains quite a bit of NaN values
df.drop(columns=['LWD'], inplace=True)
# Remove the GHI column as it is a sum of DNI and DHI
df.drop(columns=['GHI'], inplace=True)
# Drop rain_accumulation because it specifies the amount of rain since the last reset.
df.drop(columns=['rain_accumulation'], inplace=True)
# Drop rain_duration because it specifies the duration of rain, which is not useful for forecasting
df.drop(columns=['rain_duration'], inplace=True)

# drop rows where any value is missing
df.dropna(inplace=True)

df.set_index('Time(utc)', inplace=True)
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)
# Remove data from 2021-01-05 to 2021-02-23
mask = (df.index < "2021-01-05") | (df.index > "2021-02-23")
df = df[mask].copy()

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
# drop values where solar altitude is below 0
df = df[df['solar_altitude'] >= 0].copy()
# Values below 0 are set to 0 for GHI, DNI, and DHI
df[['DNI', 'DHI']] = df[['DNI', 'DHI']].clip(lower=0)
# df = df.interpolate(method="time", limit=5).copy()

# Appy min-max scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

# Save the processed DataFrame to a pickle file
df_scaled.to_pickle(FILE_PROCESSED_DTU_SOLAR_STATION)
# Save scaler
import pickle
with open(FILE_SCALARS, "wb") as f:
    pickle.dump(scaler, f)

