"""
Result analysis workflow for the Citi Bike hourly demand project.

Stages:
  1. predictions     -> score LightGBM on val/test and save row-level outputs
  2. importance      -> export LightGBM feature importance table + plot
  3. error           -> export error summary tables + diagnostic plots
  4. visualizations  -> export presentation-friendly result plots
"""

import argparse
import calendar
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
MPLCONFIGDIR = os.path.join(ROOT, "results", ".mplconfig")
CACHE_DIR = os.path.join(ROOT, "results", ".cache")
os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
TABLES_DIR = os.path.join(ANALYSIS_DIR, "tables")
PLOTS_DIR = os.path.join(ANALYSIS_DIR, "plots")

FEATURES = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
    "start_lat", "start_lng",
    "temperature", "precipitation", "snowfall", "snow_depth",
    "lag_1h", "lag_2h", "lag_24h",
]
TARGET = "departures"
WEATHER_COLS = ["temperature", "precipitation", "snowfall", "snow_depth", "is_holiday"]
PREDICTIONS_PATH = os.path.join(TABLES_DIR, "lightgbm_val_test_predictions.parquet")
PREDICTION_SUMMARY_PATH = os.path.join(TABLES_DIR, "lightgbm_prediction_summary.csv")
IMPORTANCE_PATH = os.path.join(TABLES_DIR, "lightgbm_feature_importance.csv")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_dirs() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def summarize_groups(df: pd.DataFrame, keys) -> pd.DataFrame:
    if isinstance(keys, str):
        keys = [keys]

    rows = []
    for group_values, group in df.groupby(keys, sort=True, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(keys, group_values))
        row.update({
            "count": len(group),
            "mean_actual": float(group[TARGET].mean()),
            "mean_prediction": float(group["prediction"].mean()),
            "mean_residual": float(group["residual"].mean()),
            "mae": float(group["abs_error"].mean()),
            "rmse": rmse(group[TARGET], group["prediction"]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def load_features_for_predictions() -> pd.DataFrame:
    columns = [
        "station_id", "station_name", "datetime", "split", TARGET,
        "hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
        "start_lat", "start_lng",
        "temperature", "precipitation", "snowfall", "snow_depth",
        "lag_1h", "lag_2h", "lag_24h",
    ]
    log("Loading features.parquet for val/test scoring …")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, "features.parquet"), columns=columns)
    df = df.sort_values("datetime").reset_index(drop=True)
    df[WEATHER_COLS] = df[WEATHER_COLS].ffill().fillna(0)
    df = df[df["split"].isin(["val", "test"])].copy()
    df[FEATURES] = df[FEATURES].astype("float32")
    log(f"  val/test rows loaded: {len(df):,}")
    return df


def load_model():
    model_path = os.path.join(MODELS_DIR, "lightgbm.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    return joblib.load(model_path)


def resolve_feature_names(raw_names) -> list[str]:
    resolved = []
    for name in raw_names:
        if name.startswith("Column_"):
            suffix = name.split("_", maxsplit=1)[1]
            if suffix.isdigit():
                idx = int(suffix)
                if 0 <= idx < len(FEATURES):
                    resolved.append(FEATURES[idx])
                    continue
        resolved.append(name)
    return resolved


def require_predictions() -> pd.DataFrame:
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            "Prediction table not found. Run `--stage predictions` first."
        )
    return pd.read_parquet(PREDICTIONS_PATH)


def stage_predictions() -> None:
    ensure_dirs()
    df = load_features_for_predictions()
    model = load_model()

    log("Scoring LightGBM on val/test …")
    df["prediction"] = np.clip(model.predict(df[FEATURES]), 0, None)
    df["residual"] = df[TARGET] - df["prediction"]
    df["abs_error"] = df["residual"].abs()
    df["sq_error"] = df["residual"] ** 2

    keep_cols = [
        "station_id", "station_name", "datetime", "split", TARGET, "prediction",
        "residual", "abs_error", "sq_error",
        "hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday",
        "temperature", "precipitation", "snowfall", "snow_depth",
        "lag_1h", "lag_2h", "lag_24h",
    ]
    df[keep_cols].to_parquet(PREDICTIONS_PATH, index=False)

    summary = summarize_groups(df, "split")
    overall = pd.DataFrame([{
        "split": "overall_val_test",
        "count": len(df),
        "mean_actual": float(df[TARGET].mean()),
        "mean_prediction": float(df["prediction"].mean()),
        "mean_residual": float(df["residual"].mean()),
        "mae": float(df["abs_error"].mean()),
        "rmse": rmse(df[TARGET], df["prediction"]),
    }])
    pd.concat([summary, overall], ignore_index=True).to_csv(PREDICTION_SUMMARY_PATH, index=False)

    log(f"Saved prediction table → {PREDICTIONS_PATH}")
    log(f"Saved prediction summary → {PREDICTION_SUMMARY_PATH}")


def stage_importance() -> None:
    ensure_dirs()
    model = load_model()
    booster = model.booster_

    importance = pd.DataFrame({
        "feature": resolve_feature_names(booster.feature_name()),
        "importance_gain": booster.feature_importance(importance_type="gain"),
        "importance_split": booster.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    importance["importance_pct"] = (
        100 * importance["importance_gain"] / importance["importance_gain"].sum()
    )
    importance.to_csv(IMPORTANCE_PATH, index=False)

    top = importance.head(15).sort_values("importance_gain")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance_pct"], color="#2a9d8f")
    ax.set_xlabel("Gain importance (%)")
    ax.set_title("LightGBM feature importance")
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "lightgbm_feature_importance.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    log(f"Saved feature importance table → {IMPORTANCE_PATH}")
    log(f"Saved feature importance plot → {out_path}")


def stage_error() -> None:
    ensure_dirs()
    df = require_predictions()

    log("Building error summary tables …")
    by_hour = summarize_groups(df, ["split", "hour_of_day"])
    by_hour.to_csv(os.path.join(TABLES_DIR, "error_by_hour.csv"), index=False)

    by_month = summarize_groups(df, ["split", "month"])
    by_month.to_csv(os.path.join(TABLES_DIR, "error_by_month.csv"), index=False)

    by_weekend = summarize_groups(df, ["split", "is_weekend"])
    by_weekend.to_csv(os.path.join(TABLES_DIR, "error_by_weekend.csv"), index=False)

    bins = [-0.1, 0.5, 2.5, 5.5, 10.5, 20.5, np.inf]
    labels = ["0", "1-2", "3-5", "6-10", "11-20", "21+"]
    df["demand_bucket"] = pd.cut(df[TARGET], bins=bins, labels=labels)
    by_bucket = summarize_groups(df, ["split", "demand_bucket"])
    by_bucket.to_csv(os.path.join(TABLES_DIR, "error_by_demand_bucket.csv"), index=False)

    worst_stations = (
        df[df["split"] == "test"]
        .groupby(["station_id", "station_name"], sort=False)
        .agg(
            count=(TARGET, "size"),
            mean_actual=(TARGET, "mean"),
            mean_prediction=("prediction", "mean"),
            mean_residual=("residual", "mean"),
            mae=("abs_error", "mean"),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
        .head(20)
    )
    worst_stations["rmse"] = worst_stations.apply(
        lambda row: rmse(
            df[(df["split"] == "test") & (df["station_id"] == row["station_id"])][TARGET],
            df[(df["split"] == "test") & (df["station_id"] == row["station_id"])]["prediction"],
        ),
        axis=1,
    )
    worst_stations.to_csv(os.path.join(TABLES_DIR, "worst_test_stations_by_mae.csv"), index=False)

    log("Building error plots …")
    scatter_sample = df.sample(n=min(100_000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        scatter_sample[TARGET], scatter_sample["prediction"],
        s=6, alpha=0.15, color="#264653", edgecolors="none",
    )
    max_val = float(max(scatter_sample[TARGET].max(), scatter_sample["prediction"].max()))
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="#e76f51", linewidth=1.5)
    ax.set_xlabel("Actual departures")
    ax.set_ylabel("Predicted departures")
    ax.set_title("Actual vs predicted (sample)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted_scatter.png"), dpi=200)
    plt.close(fig)

    hist_sample = df.sample(n=min(200_000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(hist_sample["residual"], bins=80, color="#457b9d", alpha=0.9)
    ax.axvline(0, color="#e63946", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution (sample)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "residual_histogram.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for split_name, group in by_hour.groupby("split", sort=False):
        ax.plot(group["hour_of_day"], group["mae"], marker="o", linewidth=2, label=split_name)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("MAE")
    ax.set_title("Error by hour of day")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "error_by_hour.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    month_plot = by_month.copy()
    month_plot["month"] = month_plot["month"].astype(int)
    month_plot["month_label"] = month_plot["month"].map(lambda m: calendar.month_abbr[m])
    month_plot = month_plot.sort_values("month")

    split_colors = {"val": "#457b9d", "test": "#e76f51"}
    bar_colors = month_plot["split"].map(split_colors).fillna("#8d99ae")
    ax.bar(month_plot["month_label"], month_plot["mae"], color=bar_colors)
    ax.set_xlabel("Month")
    ax.set_ylabel("MAE")
    ax.set_title("Validation/Test error by month (Oct–Dec 2024)")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=split_colors["val"], label="val"),
        plt.Rectangle((0, 0), 1, 1, color=split_colors["test"], label="test"),
    ]
    ax.legend(handles=legend_handles)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "error_by_month_test.png"), dpi=200)
    plt.close(fig)

    log(f"Saved error tables → {TABLES_DIR}")
    log(f"Saved error plots → {PLOTS_DIR}")


def pick_station_ids(test_df: pd.DataFrame):
    station_means = (
        test_df.groupby(["station_id", "station_name"], as_index=False)[TARGET]
        .mean()
        .rename(columns={TARGET: "mean_departures"})
    )

    picks = []
    targets = [("high", 0.90), ("mid", 0.50), ("low", 0.10)]
    used_ids = set()
    for label, quantile in targets:
        target_value = station_means["mean_departures"].quantile(quantile)
        candidates = station_means.assign(
            distance=(station_means["mean_departures"] - target_value).abs()
        ).sort_values("distance")
        for _, row in candidates.iterrows():
            if row["station_id"] not in used_ids:
                used_ids.add(row["station_id"])
                picks.append((label, row["station_id"], row["station_name"]))
                break
    return picks


def stage_visualizations() -> None:
    ensure_dirs()
    df = require_predictions()

    log("Building presentation-friendly visualizations …")
    metrics = pd.read_csv(os.path.join(RESULTS_DIR, "metrics.csv"))
    test_metrics = metrics[metrics["split"] == "test"].sort_values("rmse")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(test_metrics["model"], test_metrics["rmse"], color="#1d3557")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Model comparison on test set")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "model_comparison_test_rmse.png"), dpi=200)
    plt.close(fig)

    test_df = df[df["split"] == "test"].copy()
    station_picks = pick_station_ids(test_df)
    start_dt = pd.Timestamp("2024-12-01 00:00:00")
    end_dt = pd.Timestamp("2024-12-07 23:00:00")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (label, station_id, station_name) in zip(axes, station_picks):
        station_slice = test_df[
            (test_df["station_id"] == station_id)
            & (test_df["datetime"] >= start_dt)
            & (test_df["datetime"] <= end_dt)
        ].sort_values("datetime")
        ax.plot(station_slice["datetime"], station_slice[TARGET], label="Actual", color="#1d3557")
        ax.plot(station_slice["datetime"], station_slice["prediction"], label="Predicted", color="#e76f51")
        ax.set_title(f"{label.title()} demand station: {station_name} ({station_id})")
        ax.set_ylabel("Departures")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Datetime")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "station_timeseries_examples.png"), dpi=200)
    plt.close(fig)

    log(f"Saved visualization plots → {PLOTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Citi Bike model results.")
    parser.add_argument(
        "--stage",
        choices=["predictions", "importance", "error", "visualizations", "all"],
        default="all",
        help="Analysis stage to run.",
    )
    args = parser.parse_args()

    if args.stage in ("predictions", "all"):
        stage_predictions()
    if args.stage in ("importance", "all"):
        stage_importance()
    if args.stage in ("error", "all"):
        stage_error()
    if args.stage in ("visualizations", "all"):
        stage_visualizations()


if __name__ == "__main__":
    main()
