"""
Temporal error analysis for LightGBM_tuned.
Plots RMSE broken down by:
  1. Hour of day (test set)
  2. Day of week (test set)
  3. Month (val + test combined, Oct–Dec 2024)
Output: results/temporal_error.png
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

# ── load val + test ────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_parquet(os.path.join(PROCESSED, "features.parquet"))
weather_cols = ["temperature", "precipitation", "snowfall", "snow_depth", "is_holiday"]
df[weather_cols] = df.sort_values("datetime")[weather_cols].ffill().fillna(0)
df[FEATURES] = df[FEATURES].astype("float32")

eval_df = df[df["split"].isin(["val", "test"])].copy()
X_eval  = eval_df[FEATURES].values
y_eval  = eval_df["departures"].values.astype("float32")

model   = lgb.Booster(model_file=os.path.join(MODELS, "lightgbm_tuned.txt"))
y_pred  = np.clip(model.predict(X_eval), 0, None)

eval_df = eval_df.reset_index(drop=True)
eval_df["pred"]     = y_pred
eval_df["sq_error"] = (y_pred - y_eval) ** 2

def rmse_series(group_col, labels=None):
    grp  = eval_df.groupby(group_col)["sq_error"].mean().apply(np.sqrt)
    cnt  = eval_df.groupby(group_col)["sq_error"].count()
    if labels:
        grp.index = [labels[i] for i in grp.index]
        cnt.index = grp.index
    return grp, cnt

# ── plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Temporal Error Analysis — LightGBM_tuned",
             fontsize=13, fontweight="bold")

# ── panel 1: by hour of day ────────────────────────────────────────────────────
ax = axes[0]
rmse_h, cnt_h = rmse_series("hour_of_day")
hours = range(24)
colors_h = ["#0D47A1" if r == rmse_h.max() else
            "#90CAF9" if r == rmse_h.min() else "#2196F3"
            for r in rmse_h]
ax.bar(hours, rmse_h.values, color=colors_h, edgecolor="none", width=0.8)
ax.set_xlabel("Hour of day")
ax.set_ylabel("RMSE")
ax.set_title("RMSE by Hour of Day")
ax.set_xticks([0, 6, 12, 18, 23])
ax.axvspan(0, 5.5, alpha=0.06, color="navy", label="Night (0–5)")
ax.axvspan(7, 9.5, alpha=0.06, color="orange", label="AM peak (7–9)")
ax.axvspan(17, 19.5, alpha=0.06, color="red", label="PM peak (17–19)")
ax.legend(fontsize=7, loc="upper left")
ax.spines[["top","right"]].set_visible(False)

peak_h = int(rmse_h.idxmax())
low_h  = int(rmse_h.idxmin())
ax.annotate(f"Peak\n{rmse_h.max():.2f}", xy=(peak_h, rmse_h.max()),
            xytext=(peak_h + 1.5, rmse_h.max() + 0.1),
            fontsize=7.5, color="#0D47A1",
            arrowprops=dict(arrowstyle="->", color="#0D47A1", lw=0.8))
ax.annotate(f"Min\n{rmse_h.min():.2f}", xy=(low_h, rmse_h.min()),
            xytext=(low_h + 1.5, rmse_h.min() + 0.1),
            fontsize=7.5, color="#1565C0",
            arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.8))

# ── panel 2: by day of week ────────────────────────────────────────────────────
ax = axes[1]
dow_labels = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"}
rmse_d, cnt_d = rmse_series("day_of_week", labels=dow_labels)
colors_d = ["#E57373" if d in ["Sat","Sun"] else "#2196F3" for d in rmse_d.index]
bars = ax.bar(rmse_d.index, rmse_d.values, color=colors_d, edgecolor="none", width=0.7)
for bar, cnt in zip(bars, cnt_d.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{cnt/1000:.0f}K", ha="center", fontsize=7, color="#555")
ax.set_xlabel("Day of week")
ax.set_ylabel("RMSE")
ax.set_title("RMSE by Day of Week")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2196F3", label="Weekday"),
                   Patch(color="#E57373", label="Weekend")],
          fontsize=8)
ax.spines[["top","right"]].set_visible(False)

# ── panel 3: by month (Oct–Dec, val+test) ─────────────────────────────────────
ax = axes[2]
month_labels = {10:"Oct (val)", 11:"Nov (val)", 12:"Dec (test)"}
rmse_m, cnt_m = rmse_series("month", labels=month_labels)
colors_m = ["#90CAF9", "#90CAF9", "#0D47A1"]
bars = ax.bar(rmse_m.index, rmse_m.values, color=colors_m, edgecolor="none", width=0.5)
for bar, cnt in zip(bars, cnt_m.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{cnt/1e6:.1f}M", ha="center", fontsize=7.5, color="#555")
ax.set_xlabel("Month")
ax.set_ylabel("RMSE")
ax.set_title("RMSE by Month (Oct–Dec 2024)")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
out = os.path.join(RESULTS, "temporal_error.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

# ── print summaries ────────────────────────────────────────────────────────────
print("\nRMSE by hour of day:")
for h, r in rmse_h.items():
    print(f"  {int(h):02d}:00  {r:.4f}")

print("\nRMSE by day of week:")
for d, r in rmse_d.items():
    print(f"  {d}  {r:.4f}")

print("\nRMSE by month:")
for m, r in rmse_m.items():
    print(f"  {m}  {r:.4f}")
