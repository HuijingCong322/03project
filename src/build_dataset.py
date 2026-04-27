"""
Aggregate raw Citi Bike CSVs into hourly departure counts per station,
then merge with weather and holiday data to produce a single flat CSV.

Output: data/processed/hourly_demand.csv
Columns:
    station_id, station_name, start_lat, start_lng,
    datetime (hourly, NYC time),
    departures,
    temperature, precipitation, snowfall, snow_depth,
    is_holiday
"""

import os
import glob
import pandas as pd

ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITIBIKE_DIR = os.path.join(ROOT, "data", "raw", "citibike")
WEATHER_DIR  = os.path.join(ROOT, "data", "raw", "weather")
PROCESSED    = os.path.join(ROOT, "data", "processed")
os.makedirs(PROCESSED, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────

CITIBIKE_DTYPE = {
    "ride_id":            "str",
    "rideable_type":      "str",
    "start_station_id":   "str",
    "start_station_name": "str",
    "member_casual":      "str",
}

def load_citibike_csv(path: str) -> pd.DataFrame:
    """Load one monthly CSV, keeping only the columns we need."""
    df = pd.read_csv(
        path,
        dtype=CITIBIKE_DTYPE,
        usecols=["ride_id", "started_at",
                 "start_station_id", "start_station_name",
                 "start_lat", "start_lng"],
        parse_dates=["started_at"],
    )
    return df


# ── 1. Load & concatenate all Citi Bike CSVs ──────────────────────────────────
print("Loading Citi Bike CSVs …")
csv_paths = sorted(glob.glob(os.path.join(CITIBIKE_DIR, "*.csv")))
if not csv_paths:
    raise FileNotFoundError(
        f"No CSV files found in {CITIBIKE_DIR}. Run download_data.py first."
    )

chunks = []
for path in csv_paths:
    print(f"  {os.path.basename(path)}")
    try:
        chunks.append(load_citibike_csv(path))
    except Exception as e:
        print(f"    WARNING: could not load ({e}), skipping")

trips = pd.concat(chunks, ignore_index=True)
print(f"  Total trips loaded: {len(trips):,}")


# ── 2. Filter to Jan 2023 – Dec 2024 and drop bad rows ────────────────────────
trips["started_at"] = pd.to_datetime(trips["started_at"], utc=False, errors="coerce")
trips = trips.dropna(subset=["started_at", "start_station_id"])

start = pd.Timestamp("2023-01-01")
end   = pd.Timestamp("2024-12-31 23:59:59")
trips = trips[(trips["started_at"] >= start) & (trips["started_at"] <= end)]
print(f"  After date filter: {len(trips):,} trips")


# ── 3. Localize to NYC time ────────────────────────────────────────────────────
# Raw timestamps are already in local time (no tz info); just label them.
# ambiguous="NaT" silently marks the ~1-hour DST overlap rows as NaT;
# nonexistent="NaT" handles the spring-forward gap.  Both are then dropped.
if trips["started_at"].dt.tz is None:
    trips["started_at"] = trips["started_at"].dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    )
    n_dst = trips["started_at"].isna().sum()
    if n_dst:
        print(f"  Dropping {n_dst} rows with ambiguous/nonexistent DST times")
    trips = trips.dropna(subset=["started_at"])


# ── 4. Aggregate to hourly departure counts per station ───────────────────────
trips["hour"] = trips["started_at"].dt.floor("h")

# Station metadata: keep the most-frequent lat/lng per station_id to handle
# occasional GPS jitter in the raw data.
station_meta = (
    trips.dropna(subset=["start_lat", "start_lng"])
    .groupby(["start_station_id", "start_station_name"])[["start_lat", "start_lng"]]
    .agg(lambda x: x.mode().iloc[0])   # most common value
    .reset_index()
)

hourly = (
    trips.groupby(["start_station_id", "hour"])
    .size()
    .reset_index(name="departures")
    .rename(columns={"start_station_id": "station_id", "hour": "datetime"})
)
print(f"  Hourly aggregation: {len(hourly):,} station-hour rows")


# ── 5. Merge station metadata ─────────────────────────────────────────────────
station_meta = station_meta.rename(columns={
    "start_station_id":   "station_id",
    "start_station_name": "station_name",
})
hourly = hourly.merge(station_meta, on="station_id", how="left")


# ── 6. Merge weather ──────────────────────────────────────────────────────────
weather_path = os.path.join(WEATHER_DIR, "central_park_weather_2023_2024.csv")
weather = pd.read_csv(weather_path, parse_dates=["datetime"])
weather["datetime"] = weather["datetime"].dt.tz_localize(
    "America/New_York", ambiguous="NaT", nonexistent="NaT"
)
weather = weather.dropna(subset=["datetime"])

hourly = hourly.merge(weather, on="datetime", how="left")
missing_wx = hourly["temperature"].isna().sum()
if missing_wx:
    print(f"  WARNING: {missing_wx} rows with missing weather — forward-filling")
    hourly[["temperature","precipitation","snowfall","snow_depth"]] = (
        hourly[["temperature","precipitation","snowfall","snow_depth"]].ffill()
    )


# ── 7. Merge holidays ─────────────────────────────────────────────────────────
holiday_path = os.path.join(WEATHER_DIR, "nyc_holidays_2023_2024.csv")
if os.path.exists(holiday_path):
    holidays = pd.read_csv(holiday_path, parse_dates=["date"])
    holiday_dates = set(holidays["date"].dt.date)
    hourly["is_holiday"] = hourly["datetime"].dt.date.isin(holiday_dates).astype(int)
else:
    print("  WARNING: holiday file not found — is_holiday set to 0")
    hourly["is_holiday"] = 0


# ── 8. Final sort & save ───────────────────────────────────────────────────────
hourly = hourly.sort_values(["station_id", "datetime"]).reset_index(drop=True)

out_path = os.path.join(PROCESSED, "hourly_demand.csv")
hourly.to_csv(out_path, index=False)

print(f"\nSaved → {out_path}")
print(hourly.dtypes)
print(hourly.describe())
