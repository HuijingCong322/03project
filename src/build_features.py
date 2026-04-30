"""
Feature engineering on top of hourly_demand.csv.

Steps:
  1. Expand to a full station × hour grid (fill missing hours with 0 departures)
  2. Add time features: hour_of_day, day_of_week, month, is_weekend
  3. Add lag features per station: lag_1h, lag_2h, lag_24h
  4. Tag each row with train / val / test split
  5. Drop rows that have NaN lags (the first 24 h of each station's history)

Output: data/processed/features.parquet

LEAKAGE NOTE:
  Lags are computed by sorting the *full* dataset chronologically and then
  shifting within each station group.  A row's lag values come exclusively
  from earlier timestamps — no future information leaks in.
  The split column is added *after* lags are computed, so there is no
  separate "fit on train only" step needed here.
"""

import os
import pandas as pd
import numpy as np

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("Loading hourly_demand.csv …")
df = pd.read_csv(
    os.path.join(PROCESSED, "hourly_demand.csv"),
    dtype={"station_id": str},
    parse_dates=["datetime"],
    low_memory=False,
)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("America/New_York")
# Strip timezone — work with naive local timestamps throughout to avoid DST edge cases
df["datetime"] = df["datetime"].dt.tz_localize(None)
print(f"  {len(df):,} rows, {df['station_id'].nunique():,} stations")


# ── 2. Build full station × hour grid ─────────────────────────────────────────
# hourly_demand only contains hours with ≥1 departure.
# We fill the gaps with 0 so lag features can look back correctly.
print("Building full station × hour grid …")

hour_range = pd.date_range(
    start="2023-01-01 00:00",
    end="2024-12-31 23:00",
    freq="h",
)
stations = df[["station_id", "station_name", "start_lat", "start_lng"]].drop_duplicates("station_id")

# Cross-join stations × hours
grid = stations.assign(key=1).merge(
    pd.DataFrame({"datetime": hour_range, "key": 1}), on="key"
).drop(columns="key")

# Merge weather onto the grid (weather has one row per hour, station-independent)
weather = (
    df[["datetime", "temperature", "precipitation", "snowfall",
        "snow_depth", "is_holiday"]]
    .drop_duplicates("datetime")
    .sort_values("datetime")
)
grid = grid.merge(weather, on="datetime", how="left")

# Merge departure counts; missing = 0
demand = df[["station_id", "datetime", "departures"]]
grid = grid.merge(demand, on=["station_id", "datetime"], how="left")
grid["departures"] = grid["departures"].fillna(0).astype(np.int32)

n_total = len(stations) * len(hour_range)
print(f"  Grid: {len(grid):,} rows  (expected {n_total:,})")


# ── 3. Time features ──────────────────────────────────────────────────────────
print("Adding time features …")
grid["hour_of_day"] = grid["datetime"].dt.hour.astype(np.int8)
grid["day_of_week"]  = grid["datetime"].dt.dayofweek.astype(np.int8)   # 0=Mon
grid["month"]        = grid["datetime"].dt.month.astype(np.int8)
grid["is_weekend"]   = (grid["day_of_week"] >= 5).astype(np.int8)


# ── 4. Lag features ───────────────────────────────────────────────────────────
# Sort once; groupby preserves order within each group.
print("Computing lag features (this may take a minute) …")
grid = grid.sort_values(["station_id", "datetime"]).reset_index(drop=True)

grp = grid.groupby("station_id", sort=False)["departures"]
grid["lag_1h"]  = grp.shift(1).astype("float32")
grid["lag_2h"]  = grp.shift(2).astype("float32")
grid["lag_24h"] = grp.shift(24).astype("float32")

# Drop the first 24 hours of each station's history (lags unavailable)
before = len(grid)
grid = grid.dropna(subset=["lag_1h", "lag_2h", "lag_24h"]).reset_index(drop=True)
print(f"  Dropped {before - len(grid):,} rows with NaN lags → {len(grid):,} rows remain")


# ── 5. Train / val / test split ───────────────────────────────────────────────
# Boundaries are inclusive on the left, exclusive on the right.
dt = grid["datetime"]  # already naive local time

train_end = pd.Timestamp("2024-10-01")
val_end   = pd.Timestamp("2024-12-01")

split = pd.Series("test", index=grid.index)
split[dt < train_end]                       = "train"
split[(dt >= train_end) & (dt < val_end)]   = "val"
grid["split"] = split

counts = grid["split"].value_counts()
print(f"  train: {counts['train']:,}  val: {counts['val']:,}  test: {counts['test']:,}")


# ── 6. Save ───────────────────────────────────────────────────────────────────
out_path = os.path.join(PROCESSED, "features.parquet")
grid.to_parquet(out_path, index=False)
print(f"\nSaved → {out_path}")
print(grid.dtypes)
