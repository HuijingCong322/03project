"""
Error distribution analysis on the test set (LightGBM_tuned).
Plots:
  1. Residual histogram (prediction - actual)
  2. Actual vs predicted scatter (sampled)
  3. RMSE by demand bucket (how well does the model handle high-demand hours?)
Output: results/error_distribution.png
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
MODELS    = os.path.join(ROOT, "models")
RESULTS   = os.path.join(ROOT, "results")

FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
    "start_lat", "start_lng",
    "temperature", "precipitation", "snowfall", "snow_depth",
    "lag_1h", "lag_2h", "lag_24h",
]

# ── load test set ──────────────────────────────────────────────────────────────
print("Loading test data …")
df = pd.read_parquet(os.path.join(PROCESSED, "features.parquet"))
weather_cols = ["temperature", "precipitation", "snowfall", "snow_depth", "is_holiday"]
df[weather_cols] = df.sort_values("datetime")[weather_cols].ffill().fillna(0)
df[FEATURES] = df[FEATURES].astype("float32")
test = df[df["split"] == "test"]

X_test = test[FEATURES].values
y_test = test["departures"].values.astype("float32")

model = lgb.Booster(model_file=os.path.join(MODELS, "lightgbm_tuned.txt"))
y_pred = np.clip(model.predict(X_test), 0, None)
residuals = y_pred - y_test   # positive = overestimate, negative = underestimate

print(f"Test rows: {len(y_test):,}")
print(f"Mean residual: {residuals.mean():.4f}")
print(f"Std residual:  {residuals.std():.4f}")
print(f"% overestimate: {(residuals > 0).mean()*100:.1f}%")
print(f"% underestimate: {(residuals < 0).mean()*100:.1f}%")

# ── plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# --- panel 1: residual histogram ---
ax1 = fig.add_subplot(gs[0])
clip = 20
r_clipped = np.clip(residuals, -clip, clip)
ax1.hist(r_clipped, bins=80, color="#2196F3", edgecolor="none", alpha=0.85)
ax1.axvline(0, color="red", linewidth=1.2, linestyle="--", label="Zero error")
ax1.axvline(residuals.mean(), color="orange", linewidth=1.2,
            linestyle="-", label=f"Mean = {residuals.mean():.3f}")
ax1.set_xlabel("Residual (predicted - actual)")
ax1.set_ylabel("Count")
ax1.set_title("Residual Distribution\n(clipped to ±20)")
ax1.legend(fontsize=8)
ax1.spines[["top", "right"]].set_visible(False)

# --- panel 2: actual vs predicted scatter (sample 30k points) ---
ax2 = fig.add_subplot(gs[1])
rng = np.random.default_rng(42)
idx = rng.choice(len(y_test), size=min(30_000, len(y_test)), replace=False)
ax2.scatter(y_test[idx], y_pred[idx], alpha=0.05, s=3, color="#1565C0")
max_val = max(y_test[idx].max(), y_pred[idx].max())
ax2.plot([0, max_val], [0, max_val], "r--", linewidth=1.2, label="Perfect fit")
ax2.set_xlabel("Actual departures")
ax2.set_ylabel("Predicted departures")
ax2.set_title("Actual vs Predicted\n(30K sample, test set)")
ax2.legend(fontsize=8)
ax2.spines[["top", "right"]].set_visible(False)

# --- panel 3: RMSE by actual demand bucket ---
ax3 = fig.add_subplot(gs[2])
buckets = [0, 1, 3, 6, 10, 20, 50, 200]
labels  = ["0", "1-2", "3-5", "6-9", "10-19", "20-49", "50+"]
rmses, counts = [], []
for lo, hi, lab in zip(buckets[:-1], buckets[1:], labels):
    mask = (y_test >= lo) & (y_test < hi)
    if mask.sum() == 0:
        rmses.append(0); counts.append(0); continue
    rmse = np.sqrt(((y_pred[mask] - y_test[mask])**2).mean())
    rmses.append(rmse)
    counts.append(mask.sum())

colors = ["#90CAF9" if r < 2 else "#2196F3" if r < 4 else "#0D47A1" for r in rmses]
bars = ax3.bar(labels, rmses, color=colors, edgecolor="none")
for bar, cnt in zip(bars, counts):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"n={cnt/1000:.0f}K", ha="center", fontsize=7, color="#555")
ax3.set_xlabel("Actual departures (bucket)")
ax3.set_ylabel("RMSE")
ax3.set_title("RMSE by Demand Level")
ax3.spines[["top", "right"]].set_visible(False)

fig.suptitle("Error Distribution Analysis — LightGBM_tuned (Test Set, Dec 2024)",
             fontsize=12, fontweight="bold", y=1.02)

out = os.path.join(RESULTS, "error_distribution.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")

# ── print RMSE by bucket ───────────────────────────────────────────────────────
print("\nRMSE by demand bucket:")
for lab, rmse, cnt in zip(labels, rmses, counts):
    print(f"  departures={lab:6s}  RMSE={rmse:.4f}  n={cnt:,}")
