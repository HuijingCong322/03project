"""
Download all raw data for the Citi Bike demand prediction project:
  1. Citi Bike trip CSVs  (Jan 2023 – Dec 2024)  from S3
  2. Hourly weather       (Jan 2023 – Dec 2024)  from Open-Meteo (no key needed)
  3. NYC federal holidays (generated via `holidays` package)
"""

import os
import time
import zipfile
import requests
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITIBIKE_DIR = os.path.join(ROOT, "data", "raw", "citibike")
WEATHER_DIR  = os.path.join(ROOT, "data", "raw", "weather")

os.makedirs(CITIBIKE_DIR, exist_ok=True)
os.makedirs(WEATHER_DIR,  exist_ok=True)


# ── 1. Citi Bike trip data ─────────────────────────────────────────────────────
# URL structure (confirmed from S3 bucket listing):
#   2023 → one annual zip:   2023-citibike-tripdata.zip          (~1.5 GB)
#   2024 → monthly zips:     YYYYMM-citibike-tripdata.zip        (~370 MB – 1 GB each)
# Note: filenames have NO ".csv" component — just .zip.
CITIBIKE_BASE = "https://s3.amazonaws.com/tripdata"

def _download_zip(name: str) -> None:
    """Download a single zip from S3 and extract it into CITIBIKE_DIR."""
    url  = f"{CITIBIKE_BASE}/{name}"
    dest = os.path.join(CITIBIKE_DIR, name)

    if os.path.exists(dest):
        print(f"  [skip] {name} already downloaded")
        return

    print(f"  downloading {name} …", end=" ", flush=True)
    r = requests.get(url, timeout=600, stream=True)
    if r.status_code != 200:
        print(f"HTTP {r.status_code} — skipping")
        return

    size = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=4 << 20):   # 4 MB chunks
            f.write(chunk)
            size += len(chunk)
    print(f"done ({size / 1e6:.0f} MB)")

    print(f"  extracting {name} …", end=" ", flush=True)
    with zipfile.ZipFile(dest, "r") as zf:
        zf.extractall(CITIBIKE_DIR)
    print("done")


print("=== Citi Bike trip data ===")

# 2023: single annual file
_download_zip("2023-citibike-tripdata.zip")

# 2024: one file per month
for month in range(1, 13):
    _download_zip(f"2024{month:02d}-citibike-tripdata.zip")
    time.sleep(0.3)   # be polite to S3


# ── 2. Hourly weather from Open-Meteo archive ─────────────────────────────────
# Central Park coords:  40.7812 N, 73.9665 W
# Variables used later: temperature_2m (°F), precipitation (mm→in), snowfall,
#                       snow_depth (m→in).
WEATHER_OUT = os.path.join(WEATHER_DIR, "central_park_weather_2023_2024.csv")

def download_weather() -> None:
    if os.path.exists(WEATHER_OUT):
        print(f"  [skip] weather file already exists")
        return

    print("  fetching hourly weather from Open-Meteo …", end=" ", flush=True)
    params = {
        "latitude":            40.7812,
        "longitude":           -73.9665,
        "start_date":          "2023-01-01",
        "end_date":            "2024-12-31",
        "hourly":              "temperature_2m,precipitation,snowfall,snow_depth",
        "temperature_unit":    "fahrenheit",
        "precipitation_unit":  "inch",
        "timezone":            "America/New_York",
    }
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params, timeout=60
    )
    r.raise_for_status()
    hourly = r.json()["hourly"]

    df = pd.DataFrame({
        "datetime":    pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],       # °F
        "precipitation": hourly["precipitation"],      # inches
        "snowfall":    hourly["snowfall"],              # inches
        "snow_depth":  hourly["snow_depth"],            # inches
    })
    df.to_csv(WEATHER_OUT, index=False)
    print(f"done  ({len(df):,} rows → {WEATHER_OUT})")


print("\n=== Weather data ===")
download_weather()


# ── 3. NYC holiday calendar ────────────────────────────────────────────────────
# Requires:  pip install holidays
HOLIDAY_OUT = os.path.join(WEATHER_DIR, "nyc_holidays_2023_2024.csv")

def generate_holidays() -> None:
    if os.path.exists(HOLIDAY_OUT):
        print(f"  [skip] holiday file already exists")
        return

    try:
        import holidays as hol
    except ImportError:
        print("  `holidays` package not found — run: pip install holidays")
        return

    records = []
    for year in [2023, 2024]:
        for date, name in hol.US(state="NY", years=year).items():
            records.append({"date": date, "holiday_name": name})

    df = pd.DataFrame(records).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(HOLIDAY_OUT, index=False)
    print(f"  saved {len(df)} holiday entries → {HOLIDAY_OUT}")


print("\n=== NYC holiday calendar ===")
generate_holidays()

print("\nAll downloads complete.")
