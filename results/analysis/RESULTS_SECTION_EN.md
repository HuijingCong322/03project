# Results

## 1. Model Performance

We compared five regression models for hourly Citi Bike station-level demand forecasting: Ridge Regression, Random Forest, XGBoost, MLP, and LightGBM. Among them, **LightGBM achieved the best overall performance** on both the validation and test sets.

| Model | Val RMSE | Val MAE | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|
| LightGBM | 2.1370 | 1.1286 | 1.5034 | 0.7689 |
| XGBoost | 2.2329 | 1.1695 | 1.5359 | 0.7880 |
| MLP | 2.2446 | 1.1738 | 1.5608 | 0.7939 |
| Random Forest | 2.2881 | 1.1827 | 1.5655 | 0.8081 |
| Ridge | 2.6255 | 1.3425 | 1.7208 | 0.8399 |

The results show a clear ranking: gradient boosting models perform best, followed by MLP and Random Forest, while Ridge performs worst. This suggests that the relationship between bike demand and the available predictors is strongly **nonlinear**, making tree-based boosting methods especially effective.

Recommended figure:

- [model_comparison_test_rmse.png](plots/model_comparison_test_rmse.png)

## 2. Feature Importance

To understand what drives the best-performing model, we analyzed the built-in feature importance of LightGBM. The model relies overwhelmingly on recent demand history.

| Feature | Gain Importance (%) |
|---|---:|
| `lag_1h` | 58.96 |
| `lag_24h` | 17.06 |
| `lag_2h` | 11.99 |
| `hour_of_day` | 5.55 |
| `start_lng` | 1.77 |
| `start_lat` | 1.72 |
| `day_of_week` | 1.49 |
| `temperature` | 0.62 |
| `precipitation` | 0.34 |
| `is_weekend` | 0.26 |

Three lag features alone account for nearly **88%** of the total gain importance. This indicates that Citi Bike demand is primarily explained by:

- short-term persistence, especially demand one hour earlier
- daily recurring patterns, especially demand twenty-four hours earlier
- time-of-day effects

Weather variables such as temperature and precipitation do contribute, but their impact is much smaller than historical demand and temporal structure.

Recommended figure:

- [lightgbm_feature_importance.png](plots/lightgbm_feature_importance.png)

## 3. Prediction Bias and Overall Error Pattern

We generated row-level predictions for all validation and test observations using the trained LightGBM model. The prediction summary is shown below.

| Split | Count | Mean Actual | Mean Predicted | Mean Residual | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 3,653,207 | 2.5132 | 2.3971 | 0.1160 | 1.1286 | 2.1370 |
| Test | 1,849,476 | 1.2953 | 1.2340 | 0.0613 | 0.7689 | 1.5034 |

Because the mean prediction is lower than the mean actual value on both splits, the model shows a **slight overall underprediction bias**. This bias becomes more visible in high-demand settings, as discussed below.

Recommended figures:

- [actual_vs_predicted_scatter.png](plots/actual_vs_predicted_scatter.png)
- [residual_histogram.png](plots/residual_histogram.png)

The scatter plot shows that the model tracks low and moderate demand reasonably well, but the predictions become more dispersed for high-demand observations. The residual histogram further suggests that most errors are concentrated near zero, although the model still makes meaningful mistakes in high-variance scenarios.

## 4. Error by Time of Day

Model performance is not uniform across the day. The largest test errors occur during the late afternoon and evening peak.

| Hour of Day | Test MAE | Test RMSE |
|---:|---:|---:|
| 17 | 1.2718 | 2.3882 |
| 18 | 1.2059 | 2.2318 |
| 16 | 1.1716 | 2.0339 |
| 15 | 1.1272 | 1.9139 |
| 14 | 1.0911 | 1.8429 |

This indicates that the model struggles most during periods of concentrated and rapidly changing demand. In other words, **the busier and more volatile the hour, the harder it is to predict accurately**.

Recommended figure:

- [error_by_hour.png](plots/error_by_hour.png)

## 5. Error by Demand Level

To better understand where the model fails, we grouped test observations by actual departure volume.

| Demand Bucket | Count | Mean Actual | Mean Predicted | Mean Residual | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `0` | 1,145,273 | 0.0000 | 0.3161 | -0.3161 | 0.3161 | 0.5900 |
| `1-2` | 419,504 | 1.3168 | 1.1984 | 0.1184 | 0.8571 | 1.1783 |
| `3-5` | 163,213 | 3.7491 | 3.0687 | 0.6803 | 1.6906 | 2.0934 |
| `6-10` | 82,806 | 7.4989 | 5.7808 | 1.7181 | 2.6741 | 3.2546 |
| `11-20` | 32,913 | 13.8361 | 9.9960 | 3.8400 | 4.6602 | 5.5137 |
| `21+` | 5,767 | 26.8828 | 18.8948 | 7.9880 | 8.6189 | 10.2518 |

Two clear patterns emerge:

- The model slightly **overpredicts zero-demand hours**, with an average prediction of about `0.32` when the true demand is `0`.
- The model increasingly **underpredicts high-demand hours**, especially in the `21+` bucket, where the average prediction falls far below the true mean.

This suggests a shrinkage effect toward the mean: the model handles common and moderate demand patterns well, but it struggles to fully capture rare demand surges.

## 6. Error by Month

Monthly error patterns also help explain why the validation metrics are worse than the test metrics. Instead of plotting only the test month, we compare the full `Oct–Dec 2024` evaluation window.

| Month | Split | MAE |
|---|---|---:|
| Oct 2024 | Validation | 1.2211 |
| Nov 2024 | Validation | 1.0328 |
| Dec 2024 | Test | 0.7689 |

The figure shows a steady decline in MAE from October to December. This means the validation period was not simply worse by chance; rather, the months included in validation were genuinely harder to predict than the test month.

Recommended figure:

- [error_by_month_test.png](plots/error_by_month_test.png)

## 7. Station-Level Error

The most difficult stations are concentrated in high-traffic Manhattan locations. A few examples from the test set are shown below.

| Station | Mean Actual | Mean Predicted | MAE |
|---|---:|---:|---:|
| W 21 St & 6 Ave | 12.0202 | 11.2544 | 3.5610 |
| W 31 St & 7 Ave | 10.8024 | 9.6723 | 3.5477 |
| Broadway & E 14 St | 10.3105 | 9.2910 | 3.2300 |
| University Pl & E 14 St | 10.1586 | 9.4246 | 3.2186 |
| 9 Ave & W 33 St | 9.4745 | 8.8221 | 3.2058 |

These stations tend to have high average usage and strong short-term fluctuations, which makes them harder to model than quieter stations.

Recommended figure:

- [station_timeseries_examples.png](plots/station_timeseries_examples.png)

The time-series examples provide an intuitive comparison across high-, medium-, and low-demand stations. They show that the model tracks low- and medium-demand stations more reliably, while large peaks at busy stations remain more difficult to capture.

## 8. Main Takeaways

Overall, the results suggest that the current pipeline is effective, but its strengths and weaknesses are now clear:

- **LightGBM is the strongest model** among all methods tested.
- **Historical demand is the dominant predictive signal**, especially `lag_1h`, `lag_24h`, and `lag_2h`.
- **Time-of-day structure matters**, especially during commuting and evening peaks.
- **Errors increase sharply in high-demand, high-volatility settings**.
- **Busy Manhattan stations remain the hardest locations to predict**.

In summary, the model already captures the broad temporal structure of Citi Bike demand well, but its main remaining challenge is to improve performance in peak-demand scenarios and at highly active stations.
