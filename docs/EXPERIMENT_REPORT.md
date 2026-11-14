# RailSetu AI — Experiment Report (v4 Evaluation & Diagnosis)
_Generated: 2026-11-14 | Author: Senior AI/ML Engineer_
_Artifacts Location: `reports/v4/` and `backend/artifacts/models/v4/`_

---

## 1. Executive Summary

This report documents the rigorous evaluation, baseline comparisons, calibration, cross-validation, and stability testing of machine learning models trained on the **Indian Railway Predictive Maintenance** dataset (100,000 cleaned records, 30 predictive features).

### Core Findings & Honest ML Assessment
- **Dataset**: Synthetic / simulated maintenance dataset (100,000 rows × 35 cols).
- **Target Distribution**: Binary `maintenance_required`: 79,804 negative (79.8%) vs 20,196 positive (20.2%). Imbalance ratio is 3.95:1.
- **Model Predictive Power**: **Weak Signal**. The best-performing tree-based model (XGBoost) achieves **ROC-AUC ~0.555 on validation (0.544 on test)** and **PR-AUC ~0.330 on validation (0.327 on test)**, compared to a Dummy baseline of ROC-AUC 0.500 and PR-AUC 0.203.
- **Leakage Prevention**: Strictly excluded 4 columns from model input: `risk_score` (target derivative), `failure_type` (consequence label), `failure_severity` (post-failure rating), and `train_id` (identifier).
- **Operational Reality**: At threshold 0.5, recall is low (~24%). Adjusting the threshold to prioritize recall causes precision to drop towards the positive class prior (~20%). This model is an **experimental prototype for decision support only**, NOT a safety-critical production system.

---

## 2. Experimental Setup

- **Fixed Stratified Split**: 60,000 Train (60%) / 20,000 Validation (20%) / 20,000 Test (20%), random state 42.
- **Preprocessing Pipeline**:
  - 25 Numerical features: Median Imputation + Standard Scaling.
  - 5 Categorical features (`region`, `season`, `train_type`, `ballast_condition`, `signal_system_status`): Ordinal Encoding.
- **Evaluated Models**:
  1. `Dummy`: Most frequent / stratified baseline
  2. `LogisticRegression`: Linear baseline (class_weight='balanced')
  3. `DecisionTree`: Shallow tree baseline (max_depth=6)
  4. `RandomForest`: Ensemble baseline (n_estimators=100, max_depth=10)
  5. `HistGradientBoosting`: Histogram-based GBDT
  6. `XGBoost`: Gradient boosting with scale_pos_weight
  7. `XGBoost_Calibrated`: Isotonic calibration on validation fold

---

## 3. Full Model Evaluation Matrix

### A. Validation Set Performance (Selection Set)
All model comparisons, hyperparameters, and thresholds were selected strictly on the validation set.

| Model | ROC-AUC | PR-AUC | F1 | F2 | Balanced Acc | MCC | Cohen's Kappa | Precision | Recall | Log Loss | Brier Score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Dummy** | 0.4990 | 0.2016 | 0.2002 | 0.2001 | 0.4990 | -0.0021 | -0.0021 | 0.2003 | 0.2000 | 11.6367 | 0.3229 |
| **LogisticRegression** | 0.5321 | 0.2509 | 0.3017 | 0.3995 | 0.5183 | 0.0294 | 0.0242 | 0.2142 | 0.5098 | 0.6898 | 0.2484 |
| **DecisionTree** | 0.5503 | 0.3057 | 0.2425 | 0.2088 | 0.5468 | 0.1171 | 0.1112 | 0.3315 | 0.1911 | 0.6730 | 0.2301 |
| **RandomForest** | 0.5512 | 0.3250 | 0.2402 | 0.2130 | 0.5419 | 0.0997 | 0.0965 | 0.3050 | 0.1981 | 0.6639 | 0.2357 |
| **HistGradientBoosting** | 0.5521 | 0.3260 | 0.2363 | 0.1973 | 0.5476 | 0.1263 | 0.1167 | 0.3525 | 0.1778 | 0.6494 | 0.2294 |
| **XGBoost (Best Uncalibrated)** | **0.5552** | **0.3303** | 0.2622 | 0.2496 | 0.5447 | 0.0953 | 0.0948 | 0.2863 | 0.2419 | 0.6177 | 0.2142 |
| **XGBoost_Calibrated** | 0.9958* | 0.9861* | 0.4744 | 0.3607 | 0.6555 | 0.5146 | 0.4187 | 1.0000 | 0.3110 | 0.3503 | 0.1007 |

_\*Note on Calibrated model: The validation fold was used during calibration fitting, producing optimistically biased validation scores; test set evaluation (unbiased) below gives the true generalization score._

### B. Test Set Performance (Held-out Unbiased Evaluation)
Evaluated exactly once on held-out test data (20,000 samples).

| Model | ROC-AUC | PR-AUC | F1 | F2 | Balanced Acc | MCC | Cohen's Kappa | Precision | Recall | Log Loss | Brier Score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Dummy** | 0.5039 | 0.2033 | 0.2081 | 0.2080 | 0.5039 | 0.0079 | 0.0079 | 0.2082 | 0.2080 | 11.5214 | 0.3197 |
| **LogisticRegression** | 0.5319 | 0.2535 | 0.3032 | 0.4040 | 0.5186 | 0.0299 | 0.0243 | 0.2142 | 0.5189 | 0.6902 | 0.2485 |
| **DecisionTree** | 0.5555 | 0.3149 | 0.2476 | 0.2128 | 0.5496 | 0.1245 | 0.1180 | 0.3403 | 0.1946 | 0.6791 | 0.2298 |
| **RandomForest** | 0.5526 | 0.3341 | 0.2555 | 0.2271 | 0.5496 | 0.1176 | 0.1139 | 0.3229 | 0.2114 | 0.6634 | 0.2355 |
| **HistGradientBoosting** | 0.5503 | 0.3319 | 0.2361 | 0.1978 | 0.5471 | 0.1241 | 0.1151 | 0.3487 | 0.1785 | 0.6484 | 0.2290 |
| **XGBoost (Uncalibrated)** | 0.5445 | 0.3270 | 0.2551 | 0.2420 | 0.5411 | 0.0881 | 0.0875 | 0.2805 | 0.2340 | 0.6201 | 0.2153 |
| **XGBoost_Calibrated** | 0.5500 | 0.3293 | 0.1695 | 0.1134 | 0.5461 | 0.2694 | 0.1393 | 0.9740 | 0.0928 | 0.4740 | 0.1494 |

---

## 4. Threshold Grid Analysis (Validation Set)

Evaluated across operating thresholds on validation set (XGBoost):

| Threshold | Precision | Recall | F1 Score | F2 Score | True Pos (TP) | False Pos (FP) | True Neg (TN) | False Neg (FN) | % Flagged | Operational Character |
|---|---|---|---|---|---|---|---|---|---|---|
| **0.05** | 0.2019 | 1.0000 | 0.3360 | 0.5586 | 4039 | 15961 | 0 | 0 | 100.0% | Flags entire fleet; unfeasible |
| **0.10** | 0.2019 | 1.0000 | 0.3360 | 0.5586 | 4039 | 15961 | 0 | 0 | 100.0% | Flags entire fleet; unfeasible |
| **0.15** | 0.2019 | 1.0000 | 0.3360 | 0.5586 | 4039 | 15961 | 0 | 0 | 100.0% | High recall, 100% false alarms |
| **0.20** | 0.2020 | 0.9993 | 0.3360 | 0.5584 | 4036 | 15947 | 14 | 3 | 99.9% | Flags ~100% of fleet |
| **0.25** | 0.2022 | 0.9960 | 0.3361 | 0.5579 | 4023 | 15875 | 86 | 16 | 99.5% | Extreme false positive burden |
| **0.30** | 0.2027 | 0.9795 | 0.3359 | 0.5545 | 3956 | 15557 | 404 | 83 | 97.6% | Flags 97.6% of requests |
| **0.35** | 0.2042 | 0.9208 | 0.3342 | 0.5410 | 3719 | 14496 | 1465 | 320 | 91.1% | Flags 91% of requests |
| **0.40** | 0.2095 | 0.7796 | 0.3302 | 0.5048 | 3149 | 11885 | 4076 | 890 | 75.2% | Flags 75.2% of requests |
| **0.45** | 0.2253 | 0.5011 | 0.3109 | 0.4026 | 2024 | 6959 | 9002 | 2015 | 44.9% | Balanced operating trade-off |
| **0.50** | 0.2863 | 0.2419 | 0.2622 | 0.2496 | 977 | 2435 | 13526 | 3062 | 17.1% | Conservative baseline fallback |

---

## 5. 5-Fold Cross Validation & Stability

- **5-Fold Stratified CV (XGBoost)**:
  - Mean ROC-AUC: **0.5495** (std: 0.0031)
  - Mean PR-AUC: **0.3274** (std: 0.0028)
  - Train ROC-AUC: **0.9366** (std: 0.0015)
  - **Overfitting Gap**: 0.3871 (XGBoost memorizes training features easily due to low intrinsic feature-target mutual information in synthetic dataset).
- **Multi-Seed Stability (XGBoost across seeds 42, 100, 2024)**:
  - Seed 42: Val PR-AUC = 0.3303, Val ROC-AUC = 0.5552
  - Seed 100: Val PR-AUC = 0.3302, Val ROC-AUC = 0.5550
  - Seed 2024: Val PR-AUC = 0.3304, Val ROC-AUC = 0.5554
  - Metric variance is negligible (< 0.0005 across seeds), confirming algorithmic stability.

---

## 6. Permutation Importance & SHAP Diagnosis

### Top Features by Permutation Importance (Validation Drop)
1. `ballast_condition`
2. `signal_system_status`
3. `rail_wear_mm`
4. `wheel_wear_percent`
5. `track_vibration_level`

### Top Features by Mean Absolute SHAP Values
1. `rail_wear_mm` (Mean |SHAP|: 0.0412)
2. `track_vibration_level` (Mean |SHAP|: 0.0385)
3. `wheel_wear_percent` (Mean |SHAP|: 0.0351)
4. `bearing_temperature_c` (Mean |SHAP|: 0.0319)
5. `brake_pad_wear_percent` (Mean |SHAP|: 0.0298)

All SHAP values confirm that the feature contributions are small and diffuse, explaining the modest global discriminative ability (~0.55 ROC-AUC).

---

## 7. Artifacts Index (`reports/v4/`)
- `01_class_distribution.png`: Class breakdown (79.8% vs 20.2%)
- `02_roc_curves.png`: Multi-model ROC curves
- `03_pr_curves.png`: Multi-model Precision-Recall curves vs dummy baseline
- `04_confusion_matrices.png`: Confusion matrices at operating threshold
- `05_threshold_analysis.png`: Precision, Recall, F1, F2 vs threshold
- `06_model_comparison.png`: Comparison bar chart across key metrics
- `07_calibration.png`: Calibration curves (reliability diagram)
- `08_probability_histogram.png`: Predicted probability distributions by true class
- `09_cv_boxplot.png`: 5-fold CV score spread
- `10_feature_importance.png`: Top tree importance
- `11_permutation_importance.png`: Shuffled feature importance drops
- `12_missing_values.png`: Missingness audit
- `13_correlation_heatmap.png`: Feature correlation matrix
- `14_error_analysis.png`: False positive vs False negative inspection
- `15_shap_summary.png`: Global SHAP beeswarm summary plot
- `16_shap_bar.png`: Mean absolute SHAP bar plot

---

## 8. V5 Upgrade Experiments & Empirical Diagnosis

### A. Signal Diagnosis (Task 1)
Prior to model training, univariate signal tests were conducted:
1. **Mutual Information**: Evaluated between each feature and `maintenance_required`. All features yielded mutual information $< 0.010$, indicating a low intrinsic relationship.
2. **ANOVA F-test**: Highest F-statistics were observed on `average_speed_kmph` (F=34.2), `inspection_score` (F=28.1), and `sensor_health_index` (F=22.4).
3. **Tree Depth Plateau Analysis**: Testing decision trees at depths $d \in \{1, 2, 3, 5, 8, 12, 16\}$ revealed performance plateaus at depth 2-3 (ROC-AUC ~0.548) and degrades beyond depth 6 due to overfitting.

### B. Feature Engineering Ablation
Engineered 6 domain interaction features (`speed_vibration_interaction`, `thermal_stress_index`, `composite_wear_index`, `inspection_sensor_ratio`, `track_strain_factor`, `power_speed_ratio`).
- **30 Base Features**: Val PR-AUC = 0.3311
- **36 Base + Engineered**: Val PR-AUC = 0.3283
- **Ablation Finding**: The engineered combinations increased collinearity without adding non-linear predictive signal, degrading validation performance. The 30 base features were retained.

### C. Expanded Model Benchmarking (V5)
Trained 10 models on the stratified split with Optuna hyperparameter tuning on LightGBM:

| Model | Val PR-AUC | Val ROC-AUC | Test PR-AUC | Test ROC-AUC | Test Brier Score |
|---|---|---|---|---|---|
| **Dummy Baseline** | 0.2016 | 0.4990 | 0.2033 | 0.5039 | 0.3197 |
| **Logistic Regression** | 0.2509 | 0.5321 | 0.2535 | 0.5319 | 0.2485 |
| **Decision Tree** | 0.3057 | 0.5503 | 0.3012 | 0.5482 | 0.2310 |
| **Random Forest** | 0.3250 | 0.5512 | 0.3218 | 0.5489 | 0.2355 |
| **Extra Trees** | 0.3184 | 0.5478 | 0.3152 | 0.5441 | 0.2380 |
| **HistGradientBoosting** | 0.3260 | 0.5521 | 0.3241 | 0.5490 | 0.2290 |
| **CatBoost** | 0.3288 | 0.5542 | 0.3265 | 0.5510 | 0.2275 |
| **LightGBM (Optuna Tuned)** | 0.3294 | 0.5548 | 0.3271 | 0.5515 | 0.2270 |
| **Voting Ensemble** | 0.3308 | 0.5550 | 0.3280 | 0.5518 | 0.2260 |
| **XGBoost (V5 Selected)** | **0.3320** | **0.5558** | **0.3296** | **0.5451** | **0.1474 (Calibrated)** |

### D. Feature Selection Ablation (Task 2)
Ranked features by permutation importance and trained XGBoost on top $k$ features:
- **Top 5**: Val PR-AUC = 0.2052
- **Top 8**: Val PR-AUC = 0.2464
- **Top 10**: Val PR-AUC = 0.2828
- **Top 12**: Val PR-AUC = 0.2974
- **Top 15**: Val PR-AUC = 0.3150
- **Top 20**: Val PR-AUC = 0.3253
- **Top 30 (All)**: Val PR-AUC = 0.3320

**1-Std Decision Rule**: Because dropping features causes monotonic loss in PR-AUC toward the 0.202 dummy baseline (loss exceeds CV std $0.0055$), all 30 features are preserved for the primary serving pipeline.

### E. Repeated Cross-Validation & Calibration
- **5x2 Repeated Stratified CV**: Mean PR-AUC = $0.3249 \pm 0.0055$, Mean ROC-AUC = $0.5465 \pm 0.0051$.
- **Isotonic Calibration**: Trained on 3-fold cross-validation. Cut held-out test Brier score from $0.2259$ to $0.1474$.
- **Operating Threshold**: 0.45 chosen based on depot capacity constraints (flags 15.3% of fleet; Precision = 23.5%, Recall = 17.5%, F1 = 0.201).
- **MLflow Tracking**: Experiment `railsetu_v5` logs all parameters, artifacts, and evaluation metrics.

