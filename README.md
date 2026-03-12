# Bank Customer Churn — Binary Classification

> Calibrated stacking ensemble with Optuna HPO, pseudo-labelling, and rank-average blending to predict bank customer churn.

---

## Results

| Metric | Score      |
|---|------------|
| CV AUC (5-fold Stratified) | **~0.895** |
| Kaggle Public Score (AUC) | **0.889**  |

---


## Project Structure

```
bank-churn/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── config/
│   │   ├── preprocessing_config.yaml   # Column taxonomy, engineered features list
│   │   ├── model_config.yaml           # HPO search spaces per model
│   │   └── best_hyperparams.yaml       # Auto-generated best hyperparameters
│   ├── data/
│   │   └── data_loader.py              # CSV loading, path constants, YAML loader
│   ├── preprocessing/
│   │   ├── pipeline.py                 # 9-step preprocessing pipeline
│   │   ├── transformers.py             # Custom sklearn transformers
│   │   └── preprocessing_utils.py     # Feature engineering + SegmentStats
│   ├── models/
│   │   ├── model.py                    # Model class with pseudo-labelling workflow
│   │   ├── models_builders.py          # Pipeline, calibration, stacking factories
│   │   └── hpo_tuner.py               # Optuna HPO (maximises ROC-AUC)               
│   └── utils/
│       └── utils.py                   
├── notebooks/
│   └── eda_bank_churn.ipynb           # EDA
├── main.py                             # Entry point
└── README.md
```

---

## Approach

### 1 — Preprocessing Pipeline (9 steps)

| Step | Transformer | What it does |
|------|---|---|
| 1    | `DropIdColumns` | Removes `id`, `CustomerId`, `Surname` |
| 2    | `FinancialFeatureTransformer` | `BalanceToSalary`, `CreditScorePerAge`, `BalancePerProduct`, `BalanceIsZero` |
| 3    | `TenureFeatureTransformer` | `IsNew` (≤1y), `IsLoyal` (≥8y), `TenureGroup` (0/1/2) |
| 4    | `RiskFeatureTransformer` | `AgeRiskScore` (age band), `InactiveRichCustomer`, `EngagementScore` — balance median fitted on train only |
| 5    | `InteractionFeatureTransformer` | `Age × NumProducts`, `CreditScore × log(Balance)`, `Geography_x_Gender` |
| 6    | `SegmentStatsTransformer` | Per-Geography and per-NumOfProducts median/std stats — **fit on train only** |
| 7    | `CategoricalEncoder` | Geography → ordinal, Gender → binary, cross-feature → ordinal |
| 8    | `NumericalImputer` | Median imputation for any residual NaN |
| 9    | `StandardScaler` | Zero-mean, unit-variance normalisation |

### 2 — Calibrated Stacking Ensemble

| Model | Calibration |
|---|---|
| LightGBM | `CalibratedClassifierCV(isotonic)` |
| XGBoost | `CalibratedClassifierCV(isotonic)` |
| CatBoost | None (well-calibrated natively) |


**Meta-learner:** `LogisticRegression()` — takes `predict_proba` outputs from all 3 base models as inputs.

### 3 — Hyperparameter Optimisation

- **Optuna** with **TPE sampler** + **MedianPruner** (cuts unpromising trials early)
- **40 trials** per model, **5-fold StratifiedKFold**
- Objective: **maximise ROC-AUC**
- `scale_pos_weight` tuned via Optuna for LightGBM and XGBoost to handle class imbalance


### 4 — Rank-Average Blending

Converts each model's probability output to percentile ranks before averaging. More robust than raw probability averaging when models have different calibration scales.

```
blended = rank_average([stacking × 3, lgbm, xgb, catboost])
```

Weighting stacking 3× reflects its superior performance vs individual models.

---


## Dependencies

```
scikit-learn  xgboost  lightgbm  catboost
optuna  pandas  numpy  pyyaml  scipy
matplotlib  seaborn  shap
```

---

## EDA Highlights

Key findings from `notebooks/eda_bank_churn.ipynb`:

- ~20% churn rate → class imbalance requires `StratifiedKFold` + `scale_pos_weight`
- **Germany** shows 2× higher churn rate than France/Spain → `Geography` is a top predictor
- **Age 35–55** is the peak churn window → `AgeRiskScore` risk band feature
- **Inactive high-balance customers** have the highest individual churn rate → `InactiveRichCustomer` flag
- **NumOfProducts = 3/4** drives near-100% churn (product mis-selling signal)
- `Balance` is bimodal (many zero-balance accounts) → `BalanceIsZero` flag + log1p
