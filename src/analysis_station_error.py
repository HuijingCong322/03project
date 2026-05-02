"""
Per-station error analysis for LightGBM_tuned (test set, Dec 2024).
Plots:
  1. Top 10 highest-RMSE stations (bar chart)
  2. Top 10 lowest-RMSE stations (bar chart)
  3. RMSE vs mean demand scatter (all stations)
  4. Geographic scatter — station coords colored by RMSE
Output: results/station_error.png
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
test = df[df["split"] == "test"].copy()

X_test = test[FEATURES].values
y_test = test["departures"].values.astype("float32")

model  = lgb.Booster(model_file=os.path.join(MODELS, "lightgbm_tuned.txt"))
y_pred = np.clip(model.predict(X_test), 0, None)

test = test.reset_index(drop=True)
test["pred"]     = y_pred
test["sq_error"] = (y_pred - y_test) ** 2

# ── per-station stats ─────────────────────────────────────────────────────────
grp = test.groupby("station_id").agg(
    station_name=("station_name", "first"),
    lat=("start_lat",  "first"),
    lng=("start_lng",  "first"),
    rmse=("sq_error",  lambda x: np.sqrt(x.mean())),
    mean_demand=("departures", "mean"),
    n_hours=("departures",  "count"),
).reset_index()

# keep only stations with ≥100 test hours (avoid tiny/inactive stations)
grp = grp[grp["n_hours"] >= 100].copy()
print(f"Stations with ≥100 test hours: {len(grp)}")

top_bad  = grp.nlargest(10,  "rmse")
top_good = grp.nsmallest(10, "rmse")

# ── plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Per-Station Error Analysis — LightGBM_tuned (Test Set, Dec 2024)",
             fontsize=13, fontweight="bold")

# panel 1: top 10 worst stations
ax = axes[0, 0]
names_bad = [n[:30] for n in top_bad["station_name"]]
bars = ax.barh(names_bad[::-1], top_bad["rmse"].values[::-1],
               color="#E53935", edgecolor="none")
for bar, md in zip(bars, top_bad["mean_demand"].values[::-1]):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f"avg={md:.1f}", va="center", fontsize=7, color="#555")
ax.set_xlabel("RMSE")
ax.set_title("Top 10 Highest-Error Stations")
ax.spines[["top", "right"]].set_visible(False)

# panel 2: top 10 best stations
ax = axes[0, 1]
names_good = [n[:30] for n in top_good["station_name"]]
bars = ax.barh(names_good, top_good["rmse"].values,
               color="#43A047", edgecolor="none")
for bar, md in zip(bars, top_good["mean_demand"].values):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"avg={md:.2f}", va="center", fontsize=7, color="#555")
ax.set_xlabel("RMSE")
ax.set_title("Top 10 Lowest-Error Stations")
ax.spines[["top", "right"]].set_visible(False)

# panel 3: RMSE vs mean demand scatter
ax = axes[1, 0]
ax.scatter(grp["mean_demand"], grp["rmse"],
           alpha=0.25, s=8, color="#1565C0", linewidths=0)
# fit a simple trend line
coef = np.polyfit(grp["mean_demand"], grp["rmse"], 1)
xline = np.linspace(0, grp["mean_demand"].max(), 200)
ax.plot(xline, np.polyval(coef, xline), "r--", linewidth=1.2, label="Linear trend")
ax.set_xlabel("Mean hourly departures (Dec 2024)")
ax.set_ylabel("RMSE")
ax.set_title("RMSE vs Mean Demand (all stations)")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

# panel 4: geographic scatter
ax = axes[1, 1]
vmin, vmax = grp["rmse"].quantile(0.05), grp["rmse"].quantile(0.95)
sc = ax.scatter(grp["lng"], grp["lat"],
                c=grp["rmse"], cmap="RdYlGn_r",
                vmin=vmin, vmax=vmax,
                s=18, alpha=0.75, linewidths=0)
plt.colorbar(sc, ax=ax, label="RMSE", shrink=0.8)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Station RMSE — Geographic Distribution")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = os.path.join(RESULTS, "station_error.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

# ── print summaries ────────────────────────────────────────────────────────────
print(f"\nOverall RMSE stats across {len(grp)} stations:")
print(f"  Mean   {grp['rmse'].mean():.4f}")
print(f"  Median {grp['rmse'].median():.4f}")
print(f"  P90    {grp['rmse'].quantile(0.9):.4f}")
print(f"  Max    {grp['rmse'].max():.4f}  ({grp.loc[grp['rmse'].idxmax(),'station_name']})")
print(f"  Min    {grp['rmse'].min():.4f}  ({grp.loc[grp['rmse'].idxmin(),'station_name']})")

print("\nTop 10 highest-error stations:")
for _, row in top_bad.iterrows():
    print(f"  {row['station_name'][:40]:40s}  RMSE={row['rmse']:.3f}  avg_demand={row['mean_demand']:.2f}")

print("\nTop 10 lowest-error stations:")
for _, row in top_good.iterrows():
    print(f"  {row['station_name'][:40]:40s}  RMSE={row['rmse']:.3f}  avg_demand={row['mean_demand']:.2f}")
