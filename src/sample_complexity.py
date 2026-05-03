"""
Sample complexity analysis.
Trains each model at 6 data fractions (10 / 20 / 40 / 60 / 80 / 100 % of train set)
and records val + test RMSE.

Usage
-----
# Person A
python src/sample_complexity.py --models ridge,rf,mlp

# Person B
python src/sample_complexity.py --models xgboost,lgbm

# Generate plot from combined results (run after both halves are done)
python src/sample_complexity.py --plot-only

Results are appended to results/sample_complexity.csv so the script is
safe to interrupt and resume — already-completed (model, fraction) pairs
are skipped automatically.
"""

import argparse
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED   = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_CSV     = os.path.join(RESULTS_DIR, "sample_complexity.csv")
OUT_PNG     = os.path.join(RESULTS_DIR, "sample_complexity.png")

FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
    "start_lat", "start_lng",
    "temperature", "precipitation", "snowfall", "snow_depth",
    "lag_1h", "lag_2h", "lag_24h",
]

FRACTIONS       = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
MAX_SKLEARN     = 5_000_000   # RF and MLP cap to stay tractable
SEED            = 42


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── load data ──────────────────────────────────────────────────────────────────
def load_data():
    log("Loading features.parquet …")
    df = pd.read_parquet(os.path.join(PROCESSED, "features.parquet"))
    weather_cols = ["temperature", "precipitation", "snowfall", "snow_depth", "is_holiday"]
    df[weather_cols] = df.sort_values("datetime")[weather_cols].ffill().fillna(0)
    df[FEATURES] = df[FEATURES].astype("float32")

    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    test  = df[df["split"] == "test"]

    X_tr = train[FEATURES].values;  y_tr = train["departures"].values.astype("float32")
    X_va = val[FEATURES].values;    y_va = val["departures"].values.astype("float32")
    X_te = test[FEATURES].values;   y_te = test["departures"].values.astype("float32")
    log(f"  train {len(X_tr):,}  val {len(X_va):,}  test {len(X_te):,}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te


# ── checkpoint helpers ─────────────────────────────────────────────────────────
def load_done():
    if not os.path.exists(OUT_CSV):
        return set()
    df = pd.read_csv(OUT_CSV)
    return set(zip(df["model"], df["fraction"].round(2)))


def save_row(model_name, frac, n_rows, val_rmse, test_rmse, elapsed):
    row = pd.DataFrame([{
        "model": model_name, "fraction": round(frac, 2), "n_rows": n_rows,
        "val_rmse": round(val_rmse, 4), "test_rmse": round(test_rmse, 4),
        "elapsed_s": round(elapsed, 1),
    }])
    write_header = not os.path.exists(OUT_CSV)
    row.to_csv(OUT_CSV, mode="a", header=write_header, index=False)


# ── model trainers ─────────────────────────────────────────────────────────────
def train_ridge(X, y, X_va, y_va, X_te, y_te):
    m = Ridge(alpha=1.0)
    m.fit(X, y)
    return (rmse(y_va, np.clip(m.predict(X_va), 0, None)),
            rmse(y_te, np.clip(m.predict(X_te), 0, None)))


def train_rf(X, y, X_va, y_va, X_te, y_te):
    m = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                              random_state=SEED, n_jobs=-1)
    m.fit(X, y)
    return (rmse(y_va, np.clip(m.predict(X_va), 0, None)),
            rmse(y_te, np.clip(m.predict(X_te), 0, None)))


def train_mlp(X, y, X_va, y_va, X_te, y_te):
    m = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=50,
                     early_stopping=True, validation_fraction=0.05,
                     random_state=SEED, verbose=False)
    m.fit(X, y)
    return (rmse(y_va, np.clip(m.predict(X_va), 0, None)),
            rmse(y_te, np.clip(m.predict(X_te), 0, None)))


def train_xgb(X, y, X_va, y_va, X_te, y_te):
    dtr = xgb.DMatrix(X, label=y)
    dva = xgb.DMatrix(X_va, label=y_va)
    dte = xgb.DMatrix(X_te)
    params = {"objective": "reg:squarederror", "tree_method": "hist",
              "learning_rate": 0.05, "max_depth": 8, "subsample": 0.8,
              "colsample_bytree": 0.8, "seed": SEED, "nthread": -1, "verbosity": 0}
    res = {}
    m = xgb.train(params, dtr, num_boost_round=2000,
                  evals=[(dva, "val")], early_stopping_rounds=30,
                  evals_result=res, verbose_eval=False)
    pred_va = np.clip(m.predict(dva), 0, None)
    pred_te = np.clip(m.predict(dte), 0, None)
    return rmse(y_va, pred_va), rmse(y_te, pred_te)


def train_lgbm(X, y, X_va, y_va, X_te, y_te):
    dtr = lgb.Dataset(X, label=y)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    params = {"objective": "regression", "metric": "rmse", "verbosity": -1,
              "boosting_type": "gbdt", "n_jobs": -1, "seed": SEED,
              "num_leaves": 127, "learning_rate": 0.05,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1}
    cbs = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]
    m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=cbs)
    pred_va = np.clip(m.predict(X_va), 0, None)
    pred_te = np.clip(m.predict(X_te), 0, None)
    return rmse(y_va, pred_va), rmse(y_te, pred_te)


TRAINERS = {
    "ridge":   (train_ridge, False),   # (fn, sklearn_cap)
    "rf":      (train_rf,    True),
    "mlp":     (train_mlp,   True),
    "xgboost": (train_xgb,   False),
    "lgbm":    (train_lgbm,  False),
}


# ── main training loop ─────────────────────────────────────────────────────────
def run(models_to_run):
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data()
    done = load_done()
    rng  = np.random.default_rng(SEED)

    for name in models_to_run:
        fn, use_cap = TRAINERS[name]
        for frac in FRACTIONS:
            if (name, frac) in done:
                log(f"  skip {name} frac={frac:.0%} (already done)")
                continue

            n = int(len(X_tr) * frac)
            if use_cap:
                n = min(n, MAX_SKLEARN)

            idx = rng.choice(len(X_tr), size=n, replace=False)
            X_sub, y_sub = X_tr[idx], y_tr[idx]

            log(f"Training {name:10s}  frac={frac:.0%}  n={n:,} …")
            t0 = time.time()
            val_r, test_r = fn(X_sub, y_sub, X_va, y_va, X_te, y_te)
            elapsed = time.time() - t0

            save_row(name, frac, n, val_r, test_r, elapsed)
            log(f"  → val RMSE={val_r:.4f}  test RMSE={test_r:.4f}  ({elapsed:.0f}s)")

    log("Done.")


# ── plot ───────────────────────────────────────────────────────────────────────
def plot():
    import matplotlib.pyplot as plt

    if not os.path.exists(OUT_CSV):
        print("No results CSV found — run training first.")
        return

    df = pd.read_csv(OUT_CSV)
    models = df["model"].unique()

    COLORS = {"ridge": "#9E9E9E", "rf": "#FB8C00", "mlp": "#8E24AA",
              "xgboost": "#E53935", "lgbm": "#1E88E5"}
    LABELS = {"ridge": "Ridge", "rf": "Random Forest", "mlp": "MLP",
              "xgboost": "XGBoost", "lgbm": "LightGBM"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("Sample Complexity — RMSE vs Training Set Size",
                 fontsize=13, fontweight="bold")

    for split, ax in zip(["val", "test"], axes):
        col = f"{split}_rmse"
        for m in sorted(models):
            sub = df[df["model"] == m].sort_values("n_rows")
            color = COLORS.get(m, "#333")
            label = LABELS.get(m, m)
            ax.plot(sub["n_rows"] / 1e6, sub[col],
                    marker="o", linewidth=1.8, markersize=5,
                    color=color, label=label)

        ax.set_xlabel("Training rows (millions)")
        ax.set_ylabel("RMSE")
        ax.set_title(f"{'Validation' if split == 'val' else 'Test'} RMSE")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")

    # print table
    pivot = df.pivot_table(index=["model", "n_rows"], values=["val_rmse", "test_rmse"])
    print("\n", pivot.round(4).to_string())


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="ridge,rf,mlp,xgboost,lgbm",
                        help="Comma-separated models to train (ridge,rf,mlp,xgboost,lgbm)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training; just regenerate the plot from existing CSV")
    args = parser.parse_args()

    if not args.plot_only:
        models = [m.strip().lower() for m in args.models.split(",")]
        unknown = [m for m in models if m not in TRAINERS]
        if unknown:
            raise ValueError(f"Unknown models: {unknown}. Choose from {list(TRAINERS)}")
        run(models)

    plot()
