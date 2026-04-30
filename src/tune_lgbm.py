"""
Hyperparameter tuning for LightGBM via Optuna (Bayesian optimisation).

Strategy:
  - Each trial trains on a 5M-row subsample with early stopping → fast (~2 min/trial)
  - 30 trials total → ~1 hour search
  - Best params are then used to retrain on the full 38M train set
  - Final metrics appended to results/metrics.csv as "LightGBM_tuned"

Search space (key LightGBM knobs):
  num_leaves, learning_rate, min_child_samples,
  feature_fraction, bagging_fraction, reg_alpha, reg_lambda, max_bin
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
TARGET       = "departures"
SEED         = 42
TUNE_SAMPLE  = 5_000_000
N_TRIALS     = 30


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── load data ──────────────────────────────────────────────────────────────────
log("Loading features.parquet …")
df = pd.read_parquet(os.path.join(PROCESSED, "features.parquet"))
weather_cols = ["temperature", "precipitation", "snowfall", "snow_depth", "is_holiday"]
df[weather_cols] = df.sort_values("datetime")[weather_cols].ffill().fillna(0)
df[FEATURES] = df[FEATURES].astype("float32")

train_df = df[df["split"] == "train"]
val_df   = df[df["split"] == "val"]
test_df  = df[df["split"] == "test"]

X_train_full = train_df[FEATURES].values
y_train_full = train_df[TARGET].values.astype("float32")
X_val  = val_df[FEATURES].values;  y_val  = val_df[TARGET].values.astype("float32")
X_test = test_df[FEATURES].values; y_test = test_df[TARGET].values.astype("float32")
log(f"  full train {len(X_train_full):,}  val {len(X_val):,}  test {len(X_test):,}")

# Fixed subsample for all trials (reproducible)
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X_train_full), size=TUNE_SAMPLE, replace=False)
X_sub = X_train_full[idx];  y_sub = y_train_full[idx]
log(f"  tuning subsample: {len(X_sub):,} rows")


# ── Optuna objective ───────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = {
        "objective":         "regression",
        "metric":            "rmse",
        "verbosity":         -1,
        "boosting_type":     "gbdt",
        "n_jobs":            -1,
        "seed":              SEED,
        "num_leaves":        trial.suggest_int("num_leaves", 63, 511),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq":      1,
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 10.0),
        "max_bin":           trial.suggest_categorical("max_bin", [255, 511]),
    }

    dtrain = lgb.Dataset(X_sub,  label=y_sub)
    dval   = lgb.Dataset(X_val,  label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=-1),
    ]
    result = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    pred = np.clip(result.predict(X_val), 0, None)
    return rmse(y_val, pred)


# ── run search ─────────────────────────────────────────────────────────────────
log(f"Starting Optuna search ({N_TRIALS} trials) …")
t_search = time.time()

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

log(f"Search done in {(time.time()-t_search)/60:.1f} min")
log(f"Best val RMSE: {study.best_value:.4f}")
log(f"Best params:   {study.best_params}")

# Save best params
params_path = os.path.join(RESULTS_DIR, "lgbm_best_params.json")
with open(params_path, "w") as f:
    json.dump({"best_val_rmse": study.best_value, **study.best_params}, f, indent=2)
log(f"Saved params → {params_path}")


# ── retrain on full data with best params ──────────────────────────────────────
log("Retraining on full 38M train set with best params …")
best = study.best_params
final_params = {
    "objective":         "regression",
    "metric":            "rmse",
    "verbosity":         -1,
    "boosting_type":     "gbdt",
    "n_jobs":            -1,
    "seed":              SEED,
    **best,
    "bagging_freq":      1,   # required when bagging_fraction < 1
}

dtrain_full = lgb.Dataset(X_train_full, label=y_train_full)
dval_ds     = lgb.Dataset(X_val,        label=y_val, reference=dtrain_full)

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=100),
]
t0 = time.time()
final_model = lgb.train(
    final_params,
    dtrain_full,
    num_boost_round=3000,
    valid_sets=[dval_ds],
    callbacks=callbacks,
)
log(f"  done in {time.time()-t0:.1f}s  best_iteration={final_model.best_iteration}")


# ── evaluate ───────────────────────────────────────────────────────────────────
results = []
for split, Xs, ys in [("val", X_val, y_val), ("test", X_test, y_test)]:
    pred = np.clip(final_model.predict(Xs), 0, None)
    m = {"model": "LightGBM_tuned", "split": split,
         "rmse": rmse(ys, pred), "mae": mean_absolute_error(ys, pred)}
    results.append(m)
    log(f"  {split}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}")

final_model.save_model(os.path.join(MODELS_DIR, "lightgbm_tuned.txt"))


# ── append to metrics.csv ──────────────────────────────────────────────────────
metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
new_rows = pd.DataFrame(results)
existing = pd.read_csv(metrics_path)
existing = existing[existing["model"] != "LightGBM_tuned"]
combined = pd.concat([existing, new_rows], ignore_index=True)
combined.to_csv(metrics_path, index=False)

print("\n" + "="*60)
print(combined.pivot(index="model", columns="split", values=["rmse", "mae"])
      .round(4).sort_values(("rmse", "test")).to_string())
print("="*60)
log(f"Saved → {metrics_path}")
