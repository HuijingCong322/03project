"""
Train a LightGBM model and append its metrics to results/metrics.csv.
Uses the full 38M-row train set with early stopping on val.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED   = os.path.join(ROOT, "data", "processed")
MODELS_DIR  = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")

FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
    "start_lat", "start_lng",
    "temperature", "precipitation", "snowfall", "snow_depth",
    "lag_1h", "lag_2h", "lag_24h",
]
TARGET = "departures"
SEED   = 42


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── load ───────────────────────────────────────────────────────────────────────
log("Loading features.parquet …")
df = pd.read_parquet(os.path.join(PROCESSED, "features.parquet"))

weather_cols = ["temperature", "precipitation", "snowfall", "snow_depth", "is_holiday"]
df[weather_cols] = df.sort_values("datetime")[weather_cols].ffill().fillna(0)
df[FEATURES] = df[FEATURES].astype("float32")

train = df[df["split"] == "train"]
val   = df[df["split"] == "val"]
test  = df[df["split"] == "test"]

X_train, y_train = train[FEATURES].values, train[TARGET].values.astype("float32")
X_val,   y_val   = val[FEATURES].values,   val[TARGET].values.astype("float32")
X_test,  y_test  = test[FEATURES].values,  test[TARGET].values.astype("float32")
log(f"  train {len(X_train):,}  val {len(X_val):,}  test {len(X_test):,}")


# ── train ──────────────────────────────────────────────────────────────────────
log("Training LightGBM (full train set) …")
t0 = time.time()

model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    n_jobs=-1,
    random_state=SEED,
    verbose=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(stopping_rounds=20, verbose=True),
        lgb.log_evaluation(period=50),
    ],
)
log(f"  done in {time.time()-t0:.1f}s  best_iteration={model.best_iteration_}")


# ── evaluate ───────────────────────────────────────────────────────────────────
results = []
for split, Xs, ys in [("val", X_val, y_val), ("test", X_test, y_test)]:
    pred = np.clip(model.predict(Xs), 0, None)
    m = {"model": "LightGBM", "split": split,
         "rmse": rmse(ys, pred), "mae": mean_absolute_error(ys, pred)}
    results.append(m)
    log(f"  {split}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")


# ── save model ─────────────────────────────────────────────────────────────────
joblib.dump(model, os.path.join(MODELS_DIR, "lightgbm.joblib"))


# ── append to metrics.csv ──────────────────────────────────────────────────────
metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
new_rows = pd.DataFrame(results)

if os.path.exists(metrics_path):
    existing = pd.read_csv(metrics_path)
    existing = existing[existing["model"] != "LightGBM"]   # drop if re-running
    combined = pd.concat([existing, new_rows], ignore_index=True)
else:
    combined = new_rows

combined.to_csv(metrics_path, index=False)

print("\n" + "="*60)
print(combined.pivot(index="model", columns="split", values=["rmse", "mae"])
      .round(4).sort_values(("rmse", "test")).to_string())
print("="*60)
log(f"Saved → {metrics_path}")
