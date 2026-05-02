"""
Feature importance analysis for LightGBM_tuned (best model).
Plots both gain-based and split-based importance.
Output: results/feature_importance.png
"""

import os
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS    = os.path.join(ROOT, "results")

# Feature order matches the FEATURES list in train_lgbm.py / tune_lgbm.py
FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
    "start_lat", "start_lng",
    "temperature", "precipitation", "snowfall", "snow_depth",
    "lag_1h", "lag_2h", "lag_24h",
]
FEATURE_LABELS = {
    "lag_1h":        "Lag 1h",
    "lag_24h":       "Lag 24h",
    "lag_2h":        "Lag 2h",
    "hour_of_day":   "Hour of day",
    "temperature":   "Temperature",
    "day_of_week":   "Day of week",
    "month":         "Month",
    "start_lat":     "Latitude",
    "start_lng":     "Longitude",
    "is_weekend":    "Is weekend",
    "precipitation": "Precipitation",
    "snow_depth":    "Snow depth",
    "snowfall":      "Snowfall",
    "is_holiday":    "Is holiday",
}

model = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lightgbm_tuned.txt"))

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("LightGBM Feature Importance (tuned model)", fontsize=14, fontweight="bold")

for ax, imp_type in zip(axes, ["gain", "split"]):
    imp = model.feature_importance(importance_type=imp_type)

    # model was trained with numpy arrays → feature names are Column_N; map by index
    df = pd.DataFrame({"feature": FEATURES, "importance": imp})
    df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
    df = df.sort_values("importance", ascending=True)

    colors = ["#2196F3" if i >= len(df) - 3 else "#90CAF9" for i in range(len(df))]
    bars = ax.barh(df["label"], df["importance"], color=colors, edgecolor="none")

    # Annotate top 3
    for bar, val in zip(bars[-3:], df["importance"].values[-3:]):
        pct = val / df["importance"].sum() * 100
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=8, color="#1565C0")

    title = "Gain (contribution to loss reduction)" if imp_type == "gain" else "Split count (usage frequency)"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Importance")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"
    ))
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = os.path.join(RESULTS, "feature_importance.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

# Print table
imp_gain = model.feature_importance(importance_type="gain")
df = pd.DataFrame({"feature": FEATURES, "gain": imp_gain})
df["gain_pct"] = df["gain"] / df["gain"].sum() * 100
df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
df = df.sort_values("gain", ascending=False)
print("\nFeature importance (gain %):")
print(df[["label","gain_pct"]].to_string(index=False))
