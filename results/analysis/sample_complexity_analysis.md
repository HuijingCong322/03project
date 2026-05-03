# Sample Complexity Analysis

## Setup

Training set fractions tested: 10%, 20%, 40%, 60%, 80%, 100% of the full 38M-row train set.
Val set fixed at Oct–Nov 2024 across all runs. Models: Ridge, XGBoost, LightGBM, RandomForest.
RandomForest capped at 10M rows due to sklearn scaling constraints.

## Results

| Frac | Train rows | Ridge | XGBoost | LightGBM | RandomForest |
|------|-----------|-------|---------|----------|--------------|
| 10%  | 3.8M      | 2.626 | 2.245   | 2.172    | 2.293        |
| 20%  | 7.6M      | 2.636 | 2.239   | 2.158    | 2.280        |
| 40%  | 15.3M     | 2.667 | 2.237   | 2.145    | 2.275        |
| 60%  | 22.9M     | 2.681 | 2.234   | 2.138    | 2.275        |
| 80%  | 30.5M     | 2.691 | 2.232   | 2.139    | 2.276        |
| 100% | 38.1M     | 2.699 | 2.231   | 2.138    | 2.276        |

## Findings

### 1. LightGBM converges at 10% of training data

LightGBM Val RMSE drops from 2.172 (10%) to 2.138 (100%) — an improvement of only 0.034 across a 10x increase in training data. The curve is essentially flat from 40% onward. This means **3.8M rows already capture most of the learnable signal**; the remaining 34M rows contribute diminishing returns.

### 2. Ridge degrades slightly as training data grows (2.626 → 2.699)

Counter-intuitively, Ridge performs marginally worse with more data. The fixed regularization strength (`alpha=1.0`) provides relatively weaker constraint as the dataset grows 10x, causing the model to fit the training distribution more tightly. Combined with a mild seasonal distribution shift between train (Jan 2023–Sep 2024) and val (Oct–Nov 2024), generalization suffers slightly. The magnitude is small (~0.07 RMSE) but the direction is consistent across all fractions.

### 3. Model rankings are stable from 10% onward

LightGBM > XGBoost > RandomForest > Ridge — this ordering holds at every fraction. More data compresses the gaps between models but does not alter the ranking. XGBoost improves from 2.245 to 2.231 (0.014 gain); the improvement is real but small.

### 4. RandomForest plateaus immediately at its 10M cap

RF RMSE is essentially constant from 40%+ (2.275–2.276), confirming the 10M cap is not a meaningful constraint — the model has already saturated its capacity at that data volume.

## Key Takeaway

The performance bottleneck in this task is **not data quantity but feature expressiveness**. All models converge quickly, and none benefit substantially from scaling beyond ~10% of the training set. To meaningfully improve performance, the next step should be richer features (e.g., weather lag/lead variables, station embeddings, or cluster-level demand features) rather than collecting more data.
