"""
Naive baseline: predict departures = lag_1h.

Evaluates the simplest possible forecaster — "next hour will look like the
previous hour" — to provide a reference point for the trained models.
Appends the result to results/metrics.csv.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED   = os.path.join(ROOT, "data", "processed")
RESULTS_DIR = os.path.join(ROOT, "results")


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


print("Loading features.parquet …")
df = pd.read_parquet(os.path.join(PROCESSED, "features.parquet"))

results = []
for split in ["val", "test"]:
    sub = df[df["split"] == split]
    y = sub["departures"].values.astype("float32")
    pred = np.clip(sub["lag_1h"].values.astype("float32"), 0, None)
    r = rmse(y, pred)
    m = float(mean_absolute_error(y, pred))
    results.append({"model": "Naive (lag_1h)", "split": split, "rmse": r, "mae": m})
    print(f"  {split}  n={len(sub):,}  RMSE={r:.4f}  MAE={m:.4f}")


# ── append to metrics.csv (replace if rerun) ───────────────────────────────────
metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
new_rows = pd.DataFrame(results)

if os.path.exists(metrics_path):
    existing = pd.read_csv(metrics_path)
    existing = existing[existing["model"] != "Naive (lag_1h)"]
    combined = pd.concat([existing, new_rows], ignore_index=True)
else:
    combined = new_rows

combined.to_csv(metrics_path, index=False)
print(f"\nSaved → {metrics_path}")
print(combined.pivot(index="model", columns="split", values=["rmse", "mae"]).round(4).to_string())
