"""
Train and evaluate four regression models for Citi Bike hourly demand prediction.

Models:  Ridge Regression · Random Forest · XGBoost · MLP
Metrics: RMSE and MAE on val and test sets
Output:
  models/          – serialized model files (joblib)
  results/metrics.csv  – RMSE / MAE for every model × split

Scale note:
  Ridge and XGBoost use the full 38M-row train set.
  Random Forest and MLP are trained on a 5M-row stratified sample because
  sklearn's implementations do not scale efficiently to 38M rows.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED  = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── config ─────────────────────────────────────────────────────────────────────
FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
    "start_lat", "start_lng",
    "temperature", "precipitation", "snowfall", "snow_depth",
    "lag_1h", "lag_2h", "lag_24h",
]
TARGET = "departures"
SEED   = 42
# Max rows for models that don't scale to 38M
RF_SAMPLE  = 5_000_000
MLP_SAMPLE = 5_000_000


# ── helpers ────────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate(model, X, y, scaler=None):
    X_in = scaler.transform(X) if scaler else X
    pred = model.predict(X_in)
    pred = np.clip(pred, 0, None)   # demand cannot be negative
    return {"rmse": rmse(y, pred), "mae": mean_absolute_error(y, pred)}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── load data ──────────────────────────────────────────────────────────────────
log("Loading features.parquet …")
df = pd.read_parquet(os.path.join(PROCESSED, "features.parquet"))
# ~9 800 rows have NaN weather/holiday (DST-gap hours) — forward-fill within
# the sorted dataframe, then fill any remaining with 0 (start-of-series edge).
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

# Sampled subset for RF / MLP
rng = np.random.default_rng(SEED)
sample_idx = rng.choice(len(X_train), size=min(RF_SAMPLE, len(X_train)), replace=False)
X_train_s = X_train[sample_idx]
y_train_s = y_train[sample_idx]
log(f"  subsample for RF/MLP: {len(X_train_s):,} rows")

results = []


# ── 1. Ridge Regression ────────────────────────────────────────────────────────
log("Training Ridge …")
scaler_ridge = StandardScaler()
X_train_sc = scaler_ridge.fit_transform(X_train)

t0 = time.time()
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_sc, y_train)
log(f"  done in {time.time()-t0:.1f}s")

for split, Xs, ys in [("val", X_val, y_val), ("test", X_test, y_test)]:
    m = evaluate(ridge, Xs, ys, scaler=scaler_ridge)
    results.append({"model": "Ridge", "split": split, **m})
    log(f"  {split}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")

joblib.dump((ridge, scaler_ridge), os.path.join(MODELS_DIR, "ridge.joblib"))


# ── 2. Random Forest ───────────────────────────────────────────────────────────
log(f"Training Random Forest (n=100, subsample {RF_SAMPLE/1e6:.0f}M) …")
t0 = time.time()
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=SEED,
)
rf.fit(X_train_s, y_train_s)
log(f"  done in {time.time()-t0:.1f}s")

for split, Xs, ys in [("val", X_val, y_val), ("test", X_test, y_test)]:
    m = evaluate(rf, Xs, ys)
    results.append({"model": "RandomForest", "split": split, **m})
    log(f"  {split}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")

joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.joblib"))


# ── 3. XGBoost ─────────────────────────────────────────────────────────────────
log("Training XGBoost (full train set, tree_method=hist) …")
t0 = time.time()
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_jobs=-1,
    random_state=SEED,
    eval_metric="rmse",
    early_stopping_rounds=20,
    verbosity=1,
)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)
log(f"  done in {time.time()-t0:.1f}s  best_iteration={xgb.best_iteration}")

for split, Xs, ys in [("val", X_val, y_val), ("test", X_test, y_test)]:
    m = evaluate(xgb, Xs, ys)
    results.append({"model": "XGBoost", "split": split, **m})
    log(f"  {split}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")

xgb.save_model(os.path.join(MODELS_DIR, "xgboost.json"))


# ── 4. MLP ─────────────────────────────────────────────────────────────────────
log(f"Training MLP (subsample {MLP_SAMPLE/1e6:.0f}M) …")
scaler_mlp = StandardScaler()
X_train_mlp = scaler_mlp.fit_transform(X_train_s)

t0 = time.time()
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=1e-3,
    max_iter=50,
    early_stopping=True,
    validation_fraction=0.05,
    n_iter_no_change=5,
    random_state=SEED,
    verbose=True,
)
mlp.fit(X_train_mlp, y_train_s)
log(f"  done in {time.time()-t0:.1f}s  iterations={mlp.n_iter_}")

for split, Xs, ys in [("val", X_val, y_val), ("test", X_test, y_test)]:
    m = evaluate(mlp, Xs, ys, scaler=scaler_mlp)
    results.append({"model": "MLP", "split": split, **m})
    log(f"  {split}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")

joblib.dump((mlp, scaler_mlp), os.path.join(MODELS_DIR, "mlp.joblib"))


# ── Summary ────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
out_path = os.path.join(RESULTS_DIR, "metrics.csv")
results_df.to_csv(out_path, index=False)

print("\n" + "="*55)
print(results_df.pivot(index="model", columns="split", values=["rmse","mae"])
      .round(4).to_string())
print("="*55)
log(f"Saved → {out_path}")
