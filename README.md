# Citi Bike Hourly Demand Prediction

Predicting hourly bike departure counts per station across the NYC Citi Bike network using regression models.

## Problem

Given a station, a time slot, and current weather, how many bike departures will occur in the next hour? This is a regression task over ~1,700 active stations and 17,520 hourly time slots (Jan 2023 – Dec 2024).

## Data Sources

| Source                                                                 | Description                                                                     | Period              |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------- |
| [Citi Bike System Data](https://s3.amazonaws.com/tripdata/index.html)  | Raw trip records (ride_id, start/end station, timestamps)                       | Jan 2023 - Dec 2024 |
| [Open-Meteo Archive API](https://open-meteo.com/)                      | Hourly weather at Central Park (temp °F, precipitation, snowfall, snow depth)   | Jan 2023 - Dec 2024 |
| `holidays` (Python package)                                            | NY state federal holidays                                                       | 2023 - 2024         |

Raw data is **not committed** (≈25 GB). Download it with the script below.

## Repository Structure

```text
03project/
├── data/
│   ├── raw/
│   │   ├── citibike/          # Trip CSVs (downloaded, git-ignored)
│   │   └── weather/           # Weather + holiday CSVs (downloaded, git-ignored)
│   └── processed/
│       ├── hourly_demand.csv  # Aggregated trips + weather (git-ignored)
│       └── features.parquet   # Final feature matrix (git-ignored)
├── src/
│   ├── download_data.py       # Step 1 – download all raw data
│   ├── build_dataset.py       # Step 2 – aggregate & merge into hourly_demand.csv
│   └── build_features.py      # Step 3 – feature engineering → features.parquet
└── README.md
```

## Quickstart

### 1. Install dependencies

```bash
pip install pandas requests holidays pyarrow
```

### 2. Download raw data

```bash
python src/download_data.py
```

Downloads ~10 GB of Citi Bike trip zips from S3 and fetches weather from Open-Meteo (no API key required). The 2023 archive is a nested zip-of-zips and is handled automatically.

### 3. Build the aggregated dataset

```bash
python src/build_dataset.py
```

Produces `data/processed/hourly_demand.csv` (~18.8M station-hour rows, only hours with ≥1 departure).

### 4. Build the feature matrix

```bash
python src/build_features.py
```

Produces `data/processed/features.parquet` (~43.6M rows, full station × hour grid).

## Data Schema

### hourly_demand.csv

| Column | Type | Description |
|--------|------|-------------|
| `station_id` | str | Citi Bike station identifier |
| `station_name` | str | Human-readable station name |
| `start_lat` / `start_lng` | float | Station coordinates |
| `datetime` | datetime | Hour bucket, NYC local time |
| `departures` | int | **Target variable** — trip count in that hour |
| `temperature` | float | °F at Central Park |
| `precipitation` | float | Inches |
| `snowfall` | float | Inches |
| `snow_depth` | float | Inches |
| `is_holiday` | int | 1 if NY state holiday, else 0 |

### features.parquet (model input)

All columns above, plus:

| Column | Type | Description |
|--------|------|-------------|
| `hour_of_day` | int8 | 0–23 |
| `day_of_week` | int8 | 0 = Monday, 6 = Sunday |
| `month` | int8 | 1–12 |
| `is_weekend` | int8 | 1 if Saturday or Sunday |
| `lag_1h` | float32 | Departures at same station 1 hour prior |
| `lag_2h` | float32 | Departures at same station 2 hours prior |
| `lag_24h` | float32 | Departures at same station 24 hours prior |
| `split` | str | `train` / `val` / `test` |

Hours with zero departures are included (filled as 0) so the model learns quiet periods. The first 24 hours of each station's history are dropped (lag values unavailable).

**Leakage note:** lag features are computed by sorting the full dataset chronologically and shifting within each station group. A row's lag values come exclusively from earlier timestamps.

## Train / Val / Test Split

| Split | Period |
|-------|--------|
| Train | Jan 2023 – Sep 2024 |
| Val   | Oct – Nov 2024 |
| Test  | Dec 2024 |

## Models

Ridge Regression · Random Forest · XGBoost · MLP  
Evaluated by RMSE and MAE.

## Status

- [x] Data download pipeline
- [x] Hourly aggregation + weather/holiday merge
- [x] Feature engineering (time features + lag features)
- [ ] Model training & evaluation
- [ ] Results & analysis
