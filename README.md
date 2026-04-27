# Citi Bike Hourly Demand Prediction

Predicting hourly bike departure counts per station across the NYC Citi Bike network using regression models.

## Problem

Given a station, a time slot, and current weather, how many bike departures will occur in the next hour? This is a regression task over ~1,700 active stations and 17,520 hourly time slots (Jan 2023 – Dec 2024).

## Data Sources

| Source | Description | Period |
|--------|-------------|--------|
| [Citi Bike System Data](https://s3.amazonaws.com/tripdata/index.html) | Raw trip records (ride_id, start/end station, timestamps) | Jan 2023 – Dec 2024 |
| [Open-Meteo Archive API](https://open-meteo.com/) | Hourly weather at Central Park (temp °F, precipitation, snowfall, snow depth) | Jan 2023 – Dec 2024 |
| `holidays` (Python package) | NY state federal holidays | 2023 – 2024 |

Raw data is **not committed** (≈25 GB). Download it with the script below.

## Repository Structure

```
03project/
├── data/
│   ├── raw/
│   │   ├── citibike/        # Monthly trip CSVs (downloaded, git-ignored)
│   │   └── weather/         # Weather + holiday CSVs (downloaded, git-ignored)
│   └── processed/
│       └── hourly_demand.csv  # Aggregated output (git-ignored)
├── src/
│   ├── download_data.py     # Step 1 – download all raw data
│   └── build_dataset.py     # Step 2 – aggregate & merge into hourly_demand.csv
└── README.md
```

## Quickstart

### 1. Install dependencies

```bash
pip install pandas requests holidays
```

### 2. Download raw data

```bash
python src/download_data.py
```

Downloads ~10 GB of Citi Bike trip zips from the public S3 bucket and fetches weather from Open-Meteo (no API key required).

### 3. Build the processed dataset

```bash
python src/build_dataset.py
```

Produces `data/processed/hourly_demand.csv` with ~9.7 million station-hour rows.

### Output schema

| Column | Type | Description |
|--------|------|-------------|
| `station_id` | str | Citi Bike station identifier |
| `station_name` | str | Human-readable station name |
| `start_lat` / `start_lng` | float | Station coordinates |
| `datetime` | datetime (NYC tz) | Hour bucket (floor to hour) |
| `departures` | int | **Target variable** — trip count in that hour |
| `temperature` | float | °F at Central Park |
| `precipitation` | float | Inches |
| `snowfall` | float | Inches |
| `snow_depth` | float | Inches |
| `is_holiday` | int | 1 if NY state holiday, else 0 |

## Planned Features

- **Time:** `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_holiday`
- **Station:** latitude, longitude
- **Weather:** temperature, precipitation, snow depth
- **Lag:** demand at lag 1h, 2h, 24h per station

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
- [ ] Feature engineering
- [ ] Model training & evaluation
- [ ] Results & analysis
