"""
Actual vs Predicted visualization for LightGBM_tuned (test set, Dec 2024).
Panels:
  1. Hexbin scatter — actual vs predicted (all test rows)
  2. City-wide hourly totals — actual vs predicted time series (Dec 2024)
  3. Single busy station time series — actual vs predicted (Dec 2024)
Output: results/actual_vs_pred.png
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# ── load & predict ─────────────────────────────────────────────────────────────
print("Loading test data …")
df = pd.read_parquet(os.path.join(PROCESSED, "features.parquet"))
weather_cols = ["temperature", "precipitation", "snowfall", "snow_depth", "is_holiday"]
df[weather_cols] = df.sort_values("datetime")[weather_cols].ffill().fillna(0)
df[FEATURES] = df[FEATURES].astype("float32")
test = df[df["split"] == "test"].copy().reset_index(drop=True)

X_test = test[FEATURES].values
y_test = test["departures"].values.astype("float32")

model  = lgb.Booster(model_file=os.path.join(MODELS, "lightgbm_tuned.txt"))
y_pred = np.clip(model.predict(X_test), 0, None)
test["pred"] = y_pred

# pick a representative busy station (highest mean demand in test set)
station_rmse = (
    test.groupby(["station_id", "station_name"])
    .agg(mean_demand=("departures", "mean"), n=("departures", "count"))
    .reset_index()
)
# choose a station with high demand AND sufficient coverage
focus = station_rmse[station_rmse["n"] >= 600].nlargest(1, "mean_demand").iloc[0]
focus_id   = focus["station_id"]
focus_name = focus["station_name"]
print(f"Focus station: {focus_name}  (avg {focus['mean_demand']:.2f} dep/hr)")

# ── build city-wide hourly totals ──────────────────────────────────────────────
city = (
    test.groupby("datetime")[["departures", "pred"]]
    .sum()
    .reset_index()
    .sort_values("datetime")
)
city["datetime"] = pd.to_datetime(city["datetime"])

# ── focus station series ───────────────────────────────────────────────────────
sta = test[test["station_id"] == focus_id].sort_values("datetime").copy()
sta["datetime"] = pd.to_datetime(sta["datetime"])

# ── plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 13))
fig.suptitle("Actual vs Predicted — LightGBM_tuned (Test Set, Dec 2024)",
             fontsize=13, fontweight="bold", y=0.98)

gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)

# ── panel 1: hexbin scatter ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
cap = 30
hb = ax1.hexbin(
    np.minimum(y_test, cap),
    np.minimum(y_pred, cap),
    gridsize=60, cmap="Blues", mincnt=1, bins="log"
)
plt.colorbar(hb, ax=ax1, label="log₁₀(count)", shrink=0.85)
ax1.plot([0, cap], [0, cap], "r--", linewidth=1.2, label="Perfect fit")
ax1.set_xlabel("Actual departures (capped at 30)")
ax1.set_ylabel("Predicted departures (capped at 30)")
ax1.set_title("Hexbin: Actual vs Predicted\n(all 1.85M test rows)")
ax1.legend(fontsize=8)
ax1.spines[["top", "right"]].set_visible(False)

# ── panel 2: city-wide time series (daily aggregated for clarity) ──────────────
ax2 = fig.add_subplot(gs[0, 1])
# resample to daily totals for readability
city_daily = city.set_index("datetime").resample("D")[["departures", "pred"]].sum()
ax2.plot(city_daily.index, city_daily["departures"], color="#1565C0",
         linewidth=1.5, label="Actual")
ax2.plot(city_daily.index, city_daily["pred"],       color="#E53935",
         linewidth=1.5, linestyle="--", label="Predicted", alpha=0.85)
ax2.fill_between(city_daily.index,
                 city_daily["departures"], city_daily["pred"],
                 alpha=0.12, color="gray")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
ax2.set_ylabel("Daily departures (all stations)")
ax2.set_title("City-wide Daily Totals\n(Dec 2024)")
ax2.legend(fontsize=8)
ax2.spines[["top", "right"]].set_visible(False)

# ── panel 3: busy station — full month hourly ──────────────────────────────────
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(sta["datetime"], sta["departures"], color="#1565C0",
         linewidth=0.8, alpha=0.9, label="Actual")
ax3.plot(sta["datetime"], sta["pred"],       color="#E53935",
         linewidth=0.8, alpha=0.75, linestyle="--", label="Predicted")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax3.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
ax3.set_ylabel("Departures per hour")
ax3.set_title(f"Hourly Actual vs Predicted — {focus_name}\n(Dec 2024, avg {focus['mean_demand']:.1f} dep/hr)")
ax3.legend(fontsize=8)
ax3.spines[["top", "right"]].set_visible(False)

out = os.path.join(RESULTS, "actual_vs_pred.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

# ── print summary stats ────────────────────────────────────────────────────────
rmse_city = np.sqrt(((city_daily["departures"] - city_daily["pred"])**2).mean())
mae_city  = (city_daily["departures"] - city_daily["pred"]).abs().mean()
print(f"\nCity-wide daily totals — RMSE: {rmse_city:.0f}  MAE: {mae_city:.0f} departures/day")

rmse_sta = np.sqrt(((sta["departures"] - sta["pred"])**2).mean())
mae_sta  = (sta["departures"] - sta["pred"]).abs().mean()
print(f"Focus station ({focus_name}) — RMSE: {rmse_sta:.3f}  MAE: {mae_sta:.3f} dep/hr")
