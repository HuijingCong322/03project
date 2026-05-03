"""
Sample complexity analysis: Val RMSE vs training set size for Ridge, XGBoost,
LightGBM, and Random Forest.

Fractions tested: 10%, 20%, 40%, 60%, 80%, 100% of the full train set.
Val set is fixed across all runs. Results saved to
results/sample_complexity.csv and results/sample_complexity.png.
"""

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED   = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")

FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
    "start_lat", "start_lng",
    "temperature", "precipitation", "snowfall", "snow_depth",
    "lag_1h", "lag_2h", "lag_24h",
]
TARGET = "departures"
SEED   = 42
FRACTIONS = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

# RF is slow on large data — cap at 10M rows
RF_CAP = 10_000_000


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

train_full = df[df["split"] == "train"]
val        = df[df["split"] == "val"]

X_train_full = train_full[FEATURES].values
y_train_full = train_full[TARGET].values.astype("float32")
X_val        = val[FEATURES].values
y_val        = val[TARGET].values.astype("float32")

n_full = len(X_train_full)
log(f"  full train {n_full:,}  val {len(X_val):,}")


# ── experiment loop ────────────────────────────────────────────────────────────
records = []
rng = np.random.default_rng(SEED)

for frac in FRACTIONS:
    n = int(n_full * frac)
    idx = rng.choice(n_full, size=n, replace=False)
    X_tr = X_train_full[idx]
    y_tr = y_train_full[idx]
    log(f"\n── frac={frac:.0%}  n={n:,} ──")

    # ── Ridge ──────────────────────────────────────────────────────────────────
    log("  Ridge …")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_tr)
    t0 = time.time()
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_sc, y_tr)
    pred = np.clip(ridge.predict(scaler.transform(X_val)), 0, None)
    r = rmse(y_val, pred)
    log(f"    RMSE={r:.4f}  ({time.time()-t0:.1f}s)")
    records.append({"model": "Ridge", "frac": frac, "n_train": n, "val_rmse": r})

    # ── XGBoost ────────────────────────────────────────────────────────────────
    log("  XGBoost …")
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
        verbosity=0,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pred = np.clip(xgb.predict(X_val), 0, None)
    r = rmse(y_val, pred)
    log(f"    RMSE={r:.4f}  ({time.time()-t0:.1f}s)")
    records.append({"model": "XGBoost", "frac": frac, "n_train": n, "val_rmse": r})

    # ── LightGBM ───────────────────────────────────────────────────────────────
    log("  LightGBM …")
    t0 = time.time()
    lgbm = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        n_jobs=-1,
        random_state=SEED,
        verbose=-1,
    )
    lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    pred = np.clip(lgbm.predict(X_val), 0, None)
    r = rmse(y_val, pred)
    log(f"    RMSE={r:.4f}  ({time.time()-t0:.1f}s)")
    records.append({"model": "LightGBM", "frac": frac, "n_train": n, "val_rmse": r})

    # ── Random Forest (capped at RF_CAP) ───────────────────────────────────────
    n_rf = min(n, RF_CAP)
    idx_rf = idx[:n_rf]
    log(f"  RandomForest (n={n_rf:,}) …")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X_train_full[idx_rf], y_train_full[idx_rf])
    pred = np.clip(rf.predict(X_val), 0, None)
    r = rmse(y_val, pred)
    log(f"    RMSE={r:.4f}  ({time.time()-t0:.1f}s)")
    records.append({"model": "RandomForest", "frac": frac, "n_train": n_rf, "val_rmse": r})


# ── save CSV ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(records)
csv_path = os.path.join(RESULTS_DIR, "sample_complexity.csv")
results_df.to_csv(csv_path, index=False)
log(f"\nSaved → {csv_path}")
print(results_df.pivot(index="frac", columns="model", values="val_rmse").round(4).to_string())


# ── plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

colors  = {"Ridge": "#e15759", "XGBoost": "#f28e2b",
           "LightGBM": "#4e79a7", "RandomForest": "#59a14f"}
markers = {"Ridge": "o", "XGBoost": "s", "LightGBM": "D", "RandomForest": "^"}

for model, grp in results_df.groupby("model"):
    grp = grp.sort_values("n_train")
    ax.plot(
        grp["n_train"], grp["val_rmse"],
        marker=markers[model], color=colors[model],
        linewidth=2, markersize=7, label=model,
    )
    # annotate last point
    last = grp.iloc[-1]
    ax.annotate(
        f"{last['val_rmse']:.3f}",
        xy=(last["n_train"], last["val_rmse"]),
        xytext=(6, 0), textcoords="offset points",
        fontsize=8, color=colors[model], va="center",
    )

ax.set_xscale("log")
ax.set_xlabel("Training set size (rows, log scale)", fontsize=11)
ax.set_ylabel("Val RMSE", fontsize=11)
ax.set_title("Sample Complexity: Val RMSE vs Training Set Size", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# x-axis tick labels as "3.8M", "7.6M", etc.
xticks = results_df.groupby("model")["n_train"].apply(list).iloc[0]
ax.set_xticks(xticks)
ax.set_xticklabels([f"{x/1e6:.1f}M" for x in xticks], fontsize=9)

plt.tight_layout()
png_path = os.path.join(RESULTS_DIR, "sample_complexity.png")
plt.savefig(png_path, dpi=150)
log(f"Saved → {png_path}")
plt.show()
