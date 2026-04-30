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
├── models/                    # Serialized model files (git-ignored)
├── results/
│   └── metrics.csv            # RMSE / MAE for all models × splits
├── src/
│   ├── download_data.py       # Step 1 – download all raw data
│   ├── build_dataset.py       # Step 2 – aggregate & merge into hourly_demand.csv
│   ├── build_features.py      # Step 3 – feature engineering → features.parquet
│   └── train_models.py        # Step 4 – train & evaluate all models
└── README.md
```

## Quickstart

### 1. Install dependencies

```bash
pip install pandas requests holidays pyarrow xgboost scikit-learn joblib
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

### 5. Train and evaluate models

```bash
python src/train_models.py
```

Trains all four models and writes evaluation metrics to `results/metrics.csv`.

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

## Models and Results

Four regression models are trained and compared using RMSE and MAE.

**Training scale:** Ridge and XGBoost use the full 38M-row train set. Random Forest and MLP use a 5M-row random sample due to sklearn's scaling limitations. XGBoost uses histogram-based tree building (`tree_method=hist`) with early stopping on the validation set.

### Evaluation results

| Model        | Val RMSE | Val MAE | Test RMSE | Test MAE | Train data  |
| ------------ | -------- | ------- | --------- | -------- | ----------- |
| XGBoost *    | 2.232    | 1.172   | 1.536     | 0.788    | 38M (full)  |
| MLP          | 2.249    | 1.169   | 1.564     | 0.792    | 5M (sample) |
| RandomForest | 2.288    | 1.183   | 1.566     | 0.808    | 5M (sample) |
| Ridge        | 2.626    | 1.345   | 1.720     | 0.841    | 38M (full)  |

\* best model

**XGBoost performs best** on both val and test sets (test RMSE = 1.536, meaning average prediction error ≈ 1.5 departures per station per hour). Ridge lags behind the other three, confirming that the demand–feature relationship is non-linear. Val RMSE is higher than test RMSE across all models because Oct–Nov is the busiest autumn period with higher demand variance.

Full metrics saved in `results/metrics.csv`.

## Status

- [x] Data download pipeline (`src/download_data.py`)
- [x] Hourly aggregation + weather/holiday merge (`src/build_dataset.py`)
- [x] Feature engineering — time features + lag features (`src/build_features.py`)
- [x] Model training & evaluation — Ridge, RF, XGBoost, MLP (`src/train_models.py`)
- [ ] Results analysis — feature importance, error distribution, visualization
