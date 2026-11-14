# RailSetu AI — AI/ML Project Details
_Master file: verified facts only. Created: 2026-11-14. Updated after P2 v4 experiments._

---

## 1. DATASET

| Property | Value |
|---|---|
| **Name** | Indian Railway Predictive Maintenance |
| **Source** | Synthetic/simulated (not live Indian Railways data) |
| **Raw file (primary)** | `data/raw/indian_railway_predictive_maintenance_100k.csv` (100,300 × 35) |
| **Raw file (secondary)** | `data/raw/indian_railway_failure_detection_maintenance_v2.csv` (100,000 × 26) |
| **Cleaned file** | `data/processed/railway_cleaned.csv` (100,000 × 35) |
| **Duplicates removed** | TO VERIFY exactly; cleaned file = 100,000 rows |
| **Final rows used** | 100,000 |
| **Total columns** | 35 |
| **Feature columns used for ML** | 30 (leakage cols excluded) |

### Leakage-Excluded Columns (NOT used in any model training or inference)
| Column | Reason Excluded |
|---|---|
| `risk_score` | Derived from target — would leak ground truth |
| `failure_type` | Direct consequence of maintenance_required — target leakage |
| `failure_severity` | Same — derived post-failure label |
| `train_id` | Identifier — no predictive signal |

### Target Variable
| Property | Value |
|---|---|
| **Column** | `maintenance_required` |
| **Type** | Binary (0/1) |
| **Definition** | 1 = maintenance is required; 0 = no maintenance required |
| **Negative (0)** | 79,804 (79.8%) |
| **Positive (1)** | 20,196 (20.2%) |
| **Imbalance ratio** | 3.95:1 (negative:positive) |
| **Note** | An all-negative dummy classifier gets ~79.8% accuracy — accuracy is misleading here |

### Why accuracy is misleading (explained simply)
> Agar model sirf "No maintenance" bol de har baar, toh bhi 80% accuracy milegi. Isliye hum ROC-AUC aur PR-AUC dekhte hain — jo batata hai ki model kuch seekha hai ya nahi.

---

## 2. ALL 30 FEATURES

| # | Feature | Type | Meaning |
|---|---------|------|---------|
| 1 | `region` | Categorical | Geographical zone (North/South/East/West/Central) |
| 2 | `season` | Categorical | Monsoon/Winter/Summer/Pre-Monsoon |
| 3 | `train_type` | Categorical | Express/Passenger/Freight/Suburban |
| 4 | `train_age_years` | Numeric | Age of train in years |
| 5 | `average_speed_kmph` | Numeric | Average operating speed |
| 6 | `distance_travelled_km` | Numeric | Total distance since last maintenance |
| 7 | `track_temperature_c` | Numeric | Rail surface temperature (°C) |
| 8 | `rail_wear_mm` | Numeric | Rail head wear in millimetres — **top permutation importance feature** |
| 9 | `track_vibration_level` | Numeric | Vibration severity (simulated sensor) |
| 10 | `ballast_condition` | Categorical | Good/Fair/Poor — track bed condition |
| 11 | `track_curvature_degree` | Numeric | Track curve angle |
| 12 | `ambient_temperature_c` | Numeric | Air temperature (°C) |
| 13 | `humidity_percent` | Numeric | Relative humidity (%) |
| 14 | `rainfall_mm` | Numeric | Rainfall in mm |
| 15 | `wind_speed_kmph` | Numeric | Wind speed |
| 16 | `wheel_wear_percent` | Numeric | Wheel wear as % of tolerance — **2nd permutation importance** |
| 17 | `axle_temperature_c` | Numeric | Axle bearing temperature |
| 18 | `brake_pressure_psi` | Numeric | Brake system pressure |
| 19 | `brake_pad_wear_percent` | Numeric | Brake pad wear (%) |
| 20 | `bearing_temperature_c` | Numeric | Bearing temperature |
| 21 | `battery_voltage` | Numeric | Battery voltage (V) |
| 22 | `traction_motor_temp_c` | Numeric | Traction motor temperature |
| 23 | `signal_system_status` | Categorical | OK/Fault/Degraded |
| 24 | `power_consumption_kw` | Numeric | Power drawn (kW) |
| 25 | `load_factor_percent` | Numeric | Load as % of capacity |
| 26 | `daily_trips` | Numeric | Trips per day |
| 27 | `delay_minutes` | Numeric | Average delay per trip |
| 28 | `last_maintenance_days` | Numeric | Days since last maintenance |
| 29 | `inspection_score` | Numeric | Last inspection score (0–100; lower = worse) |
| 30 | `sensor_health_index` | Numeric | Overall sensor reliability (0–100) |

### Missing values: To be confirmed from cleaned data (expected ~0% per column in cleaned CSV based on preprocessing)

---

## 3. TRAIN / VAL / TEST SPLIT

| Split | Rows | Positive Rate | Method |
|---|---|---|---|
| Train | 60,000 | 20.2% | Stratified random split |
| Validation | 20,000 | 20.2% | Stratified random split |
| Test | 20,000 | 20.2% | Stratified random split (held out until final eval) |
| **Seed** | **42** | sklearn `train_test_split` | First 80/20 (trainval/test), then 75/25 (train/val) |

---

## 4. PREPROCESSING PIPELINE

Applied **identically** to train, val, and test (fitted on train only):

| Step | Numeric Features | Categorical Features |
|---|---|---|
| Imputation | `SimpleImputer(strategy="median")` | `SimpleImputer(strategy="most_frequent")` |
| Encoding | `StandardScaler()` | `OrdinalEncoder(unknown=-1)` |

The same `sklearn.pipeline.Pipeline` + `ColumnTransformer` is fitted on train and applied at inference — **no train/inference inconsistency** (verified: v4 model is a full Pipeline including preprocessor).

---

## 5. MODEL EXPERIMENTS — v4 (Reproduced 2026-11-14)

MLflow Experiment: `railsetu_v4` | DB: `mlflow.db`

### 5.1 All Models Summary (Validation Set, Threshold=0.5)

| Model | ROC-AUC | PR-AUC | F1 | Balanced Acc | MCC | MLflow Run ID |
|---|---|---|---|---|---|---|
| Dummy (stratified) | 0.4990 | 0.2016 | 0.200 | 0.499 | -0.002 | `9e6598db` |
| Logistic Regression | 0.5321 | 0.2509 | 0.302 | 0.518 | 0.029 | `20f4feb1` |
| Decision Tree | 0.5503 | 0.3057 | 0.243 | 0.547 | 0.117 | `30bab2a3` |
| Random Forest | 0.5512 | 0.3250 | 0.240 | 0.542 | 0.100 | `c11fb5da` |
| HistGradientBoosting | 0.5521 | 0.3260 | 0.236 | 0.548 | 0.126 | `414abe43` |
| **XGBoost (best)** | **0.5552** | **0.3303** | **0.262** | **0.545** | **0.095** | `c48238df` |
| XGBoost Calibrated* | 0.9958* | 0.9861* | 0.474* | 0.655* | 0.515* | — |

> ⚠️ *XGBoost Calibrated shows inflated metrics because `cv=3` was refitted on a subset of val — this is **overfitting to validation set**, not generalizable. Calibration is NOT reliable here and must NOT be deployed without proper held-out calibration data. Marked as unreliable in model card.

### 5.2 Hyperparameters (v4 XGBoost — Best Model)

| Parameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `scale_pos_weight` | 3.95 (class imbalance ratio) |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `eval_metric` | logloss |
| `random_state` | 42 |

### 5.3 Cross-Validation (5-Fold Stratified, XGBoost)

| Metric | Mean | Std |
|---|---|---|
| ROC-AUC | 0.5495 | ±0.0026 |
| PR-AUC | 0.3274 | ±0.0045 |
| F1 | (computed) | — |
| Balanced Accuracy | (computed) | — |
| **Overfitting gap (train-val ROC-AUC)** | **0.3871** | — |

> Large overfitting gap (0.39) on ROC-AUC means the model memorises training data but fails to generalise — consistent with weak signal in this synthetic dataset.

### 5.4 Multi-Seed Stability (5 Seeds: 42,123,456,789,1337)

| Metric | Min | Max | Range |
|---|---|---|---|
| Val ROC-AUC | 0.5408 | 0.5552 | 0.0144 |
| Val PR-AUC | 0.3195 | 0.3303 | 0.0108 |

> Low variance = model is stable, but stably mediocre.

### 5.5 Full Metric Set — XGBoost v4 at Threshold=0.5

| Metric | Train | Validation | Test |
|---|---|---|---|
| Accuracy | (see CSV) | ~0.55 | (see CSV) |
| Precision | — | 0.278 | — |
| Recall | — | 0.258 | — |
| F1 | — | 0.262 | — |
| F2 | — | 0.262 | — |
| ROC-AUC | — | 0.5552 | — |
| PR-AUC | — | 0.3303 | — |
| Balanced Accuracy | — | 0.5447 | — |
| MCC | — | 0.095 | — |
| Cohen's Kappa | — | (see CSV) | — |
| Log Loss | — | (see CSV) | — |
| Brier Score | — | (see CSV) | — |

> **Full numbers**: `reports/v4/master_metrics.csv` and `val_model_comparison.csv`

---

## 6. THRESHOLD ANALYSIS (Validation Set, XGBoost v4)

> See full table: `reports/v4/threshold_analysis.csv`

| Threshold | Precision | Recall | F1 | F2 | %Flagged | Operational Note |
|---|---|---|---|---|---|---|
| 0.05 | 0.202 | 1.000 | 0.336 | 0.559 | 100% | Flags everything — useless |
| 0.10 | ~0.202 | ~1.000 | ~0.336 | ~0.559 | ~99% | Still flags almost everything |
| 0.15 | — | — | — | — | — | See CSV |
| 0.20 | — | — | — | — | — | See CSV |
| 0.25 | — | — | — | — | — | See CSV |
| **0.30** | — | — | — | — | — | Moderate trade-off |
| 0.40 | — | — | — | — | — | Conservative |
| **0.50** | 0.278 | 0.258 | 0.262 | 0.262 | ~26% | Low recall, misses many |

**Chosen threshold: 0.05** (best val F2) — but this flags ~100% of records, confirming the model has near-zero discriminative power at any threshold that maximises recall.

> **Honest conclusion**: The model cannot reliably separate maintenance-required from non-required records. The best threshold flags everything (100% recall, 20% precision = same as the base rate). This is the key finding of the diagnostic.

---

## 7. PERMUTATION IMPORTANCE (top 5 on validation, PR-AUC drop)

| Rank | Feature | Mean Importance | Std |
|---|---|---|---|
| 1 | `rail_wear_mm` | 0.131 | ±0.002 |
| 2 | `wheel_wear_percent` | 0.117 | ±0.006 |
| 3 | `track_vibration_level` | 0.103 | ±0.005 |
| 4 | `ballast_condition` | 0.092 | ±0.005 |
| 5 | `battery_voltage` | 0.078 | ±0.003 |

> Features physically related to wear and mechanical stress are top contributors — this is domain-plausible, suggesting weak but real signal from physical features.

---

## 8. EXPLAINABILITY (SHAP)

- SHAP values computed on 2,000 val samples using `shap.Explainer(xgb_model)`
- Full files: `reports/v4/shap_feature_importance_v4.csv`, `reports/v4/15_shap_summary.png`, `reports/v4/16_shap_bar.png`
- 3 worked examples: `reports/v4/shap_worked_examples.json`
- `/ml/predict` returns top-5 SHAP contributors for single-record predictions (best-effort)

---

## 9. THREE CONCEPTS — CLEARLY SEPARATED

| Concept | Name | Source | Range | API Endpoint |
|---|---|---|---|---|
| **A** | ML Prediction Probability | XGBoost model output | 0–100% | `/api/ml/predict` |
| **B** | Dataset Risk Score | `risk_scoring.py` — derived column in cleaned CSV using condition features + pre-computed percentiles | 0–100 | Column in data only (not served directly) |
| **C** | Operational Priority Score | `priority_engine.py` — rule-based: 70% `operational_priority_score` (from simulation) + 30% deadline urgency | 0–100 | `/api/risk/score`, `/api/maintenance/priority` |

> These three are distinct and must not be conflated. Concept C is NOT ML — it is a rule-based weighted formula.

### Risk Service Formula (Concept C, in `/api/risk/score`):
```
operational_priority_score = 0.40 × ML_probability_percent + 0.60 × condition_score
```
- `condition_score` = weighted percentile rank of physical condition features (rail_wear, wheel_wear, etc.)
- `ML_probability_percent` = Concept A × 100

### Priority Engine Formula (priority_engine.py):
```
final_priority_score = 0.70 × operational_priority_score + 0.30 × deadline_urgency_score
```
- `deadline_urgency_score`: 100 if overdue, 95 if ≤1 day, 85 if ≤2 days, 75 if ≤3 days, 60 if ≤5 days, 40 if ≤7 days, 20 otherwise
- Levels: Critical (≥85), High (≥70), Medium (≥40), Low (<40)

---

## 10. HONEST MODEL ASSESSMENT

> **Summary**: The XGBoost model has a ROC-AUC of ~0.55 — just 0.05 above a random classifier (0.50). The PR-AUC of 0.33 is better than the dummy baseline of 0.20, but the absolute improvement is small.

**Reasons for weak performance** (synthetic dataset):
1. The `maintenance_required` target may have been generated with limited dependence on the 30 features, or with high noise.
2. The features are simulated sensor readings — correlations with the target may be artificially weak.
3. No feature engineering (interaction terms, ratios) was tried in v4 — v3 interaction experiment also showed no improvement.
4. The overfitting gap (0.39) is large, suggesting the model partially memorises training patterns that don't generalise.

**What the model CAN do**:
- It learns real wear-related features (rail_wear_mm, wheel_wear_percent) are more predictive — domain-plausible.
- It is better than purely random (ROC-AUC 0.55 vs 0.50).
- Permutation importance is physically interpretable.

**What the model CANNOT do**:
- Reliably identify which individual records need maintenance.
- Any threshold that catches most true faults also flags most non-fault records.
- This is NOT suitable for operational deployment or safety-critical decisions.

> Yeh model ek proof-of-concept hai — ek student project ke liye theek hai, lekin real railway safety ke liye bilkul nahi. Synthetic data pe train hua hai.

---

## 11. PLOTS GENERATED (reports/v4/)

| File | Description |
|---|---|
| `01_class_distribution.png` | Bar chart: 79,804 negative vs 20,196 positive |
| `02_roc_curves.png` | ROC curves for all 7 models on validation set |
| `03_pr_curves.png` | Precision-Recall curves (shows near-baseline PR-AUC) |
| `04_confusion_matrices.png` | Confusion matrix grid for all models at threshold=0.5 |
| `05_threshold_analysis.png` | Precision/Recall/F1/F2 vs threshold + FP-FN trade-off |
| `06_model_comparison.png` | Horizontal bar chart: val PR-AUC all models |
| `07_calibration.png` | Calibration reliability diagram (NOTE: calibrated model overfits val) |
| `08_probability_histogram.png` | Probability distribution by true class |
| `09_cv_boxplot.png` | 5-fold CV score distribution boxes |
| `10_feature_importance.png` | XGBoost internal feature importance (gain) |
| `11_permutation_importance.png` | Permutation importance (PR-AUC drop) with error bars |
| `12_missing_values.png` | % missing per feature |
| `13_correlation_heatmap.png` | Correlation heatmap (top 20 numeric features) |
| `14_error_analysis.png` | TP/FP/TN/FN pie chart at chosen threshold |
| `15_shap_summary.png` | SHAP beeswarm plot (2000 val samples) |
| `16_shap_bar.png` | SHAP mean absolute value bar chart |

---

## 12. REPRODUCIBILITY

```bash
# From project root, activate venv:
source .venv/bin/activate

# Full v4 training + evaluation + all plots:
MPLCONFIGDIR=/tmp/mpl_cache MLFLOW_DISABLE_AGENT_HINT=1 python backend/ml/train_v4.py

# View MLflow UI:
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# → open http://localhost:5000, Experiment: railsetu_v4
```

| Library | Version |
|---|---|
| Python | 3.13.0 |
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| shap | 0.54.1 |
| mlflow | 3.18.0 |
| pandas | (in .venv) |
| numpy | (in .venv) |

---

## 13. VIVA SUMMARY (1-Page)

**Project**: RailSetu AI — Railway Predictive Maintenance + Block Planning  
**Data**: 100,000 synthetic railway records, 30 features, 20% positive class (maintenance needed)  
**Task**: Binary classification — predict if maintenance is required  
**Best model**: XGBoost with class weights (scale_pos_weight=3.95)  
**Key metric**: PR-AUC (because class-imbalanced; accuracy misleads)  
**Result**: ROC-AUC=0.55 (barely above random 0.50), PR-AUC=0.33 (Dummy=0.20)  
**Honest finding**: Model has weak signal — synthetic data limitation  
**CV**: 5-fold, ROC-AUC=0.5495±0.0026 (stable but weak)  
**Top features**: rail_wear_mm, wheel_wear_percent, track_vibration_level (by permutation importance)  
**Explainability**: SHAP values confirm wear-related features drive predictions  
**Optimizer**: OR-Tools CP-SAT (Constraint Programming) — NOT ILP/MILP/GA  
**Result**: 275/500 requests scheduled (55%), all 8 validation checks PASS  

---

## 14. 15 LIKELY AI/ML VIVA Q&A

**Q1. Why did you use XGBoost over Random Forest?**  
A: XGBoost had the highest validation PR-AUC (0.33 vs 0.32 for RF). Both are ensemble tree methods, but XGBoost uses gradient boosting (sequential trees correcting errors) while RF uses bagging (parallel trees). XGBoost also has built-in class weight handling via `scale_pos_weight`. Simple English: XGBoost ek ke baad ek tree banata hai, pichle tree ki galatiyon ko theek karta hai.

**Q2. Why is accuracy misleading here?**  
A: 79.8% of records are negative. A model that always predicts "no maintenance" gets 79.8% accuracy but zero recall — it misses every actual fault. We use PR-AUC because it focuses on how well the model finds the positive class (the minority). Hinglish: Agar model hamesha "No" bole toh bhi 80% accuracy milegi, lekin koi kaam ka nahi.

**Q3. What is ROC-AUC=0.55 telling you?**  
A: Random classifier gets 0.50. Our model gets 0.55 — only 5 percentage points better than guessing randomly. This means the model has very weak discriminative power. The 5-fold CV confirmed this (0.5495±0.0026) — it's consistent but consistently weak.

**Q4. What is the difference between PR-AUC and ROC-AUC?**  
A: ROC-AUC measures discrimination across all thresholds, treating both classes equally. PR-AUC focuses on the positive class precision/recall trade-off. With 20% positives, PR-AUC is more informative — a perfect classifier gets PR-AUC=1.0, a dummy classifier gets PR-AUC=0.20 (the class rate).

**Q5. Why is the calibration suspicious (PR-AUC=0.986)?**  
A: We used `CalibratedClassifierCV(cv=3)` refitted on the validation set — this essentially memorises validation patterns and overfits. A reliable calibration requires separate held-out data. We flag this clearly and do NOT deploy the calibrated model.

**Q6. What is permutation importance?**  
A: We shuffle each feature randomly and measure how much the model's PR-AUC drops. If a feature is important, shuffling it hurts performance. `rail_wear_mm` caused the biggest drop (0.131) — it's the most important feature. Simple: Ek feature ko randomly shuffle karo aur dekho score kitna girta hai.

**Q7. Why did you choose the threshold based on F2 rather than F1?**  
A: F2 weights recall twice as much as precision. In maintenance, missing a fault (False Negative) is worse than a false alarm (False Positive) — a missed fault could cause equipment failure, while a false alarm wastes inspection effort. F2 biases towards catching more true faults.

**Q8. What is SHAP and why do you use it?**  
A: SHAP (SHapley Additive exPlanations) assigns each feature a contribution value to the prediction using game theory (Shapley values). It answers: "how much did each feature push this prediction up or down?" It makes the black-box model explainable. Hinglish: SHAP batata hai ki kaunse feature ne prediction ko kitna push kiya.

**Q9. Why is the overfitting gap 0.39?**  
A: Train ROC-AUC was ~0.94; val ROC-AUC was 0.55. This huge gap means the model memorises training data perfectly but fails to generalise — a hallmark of synthetic data with weak feature-target relationships and many trees with deep splits.

**Q10. What is a Dummy Classifier and why do you compare against it?**  
A: A Dummy Classifier predicts randomly based on class probabilities (no learning). It is the absolute baseline — a model must be better than this to have any value. Our XGBoost beats Dummy (0.33 vs 0.20 PR-AUC), but not by much.

**Q11. What algorithm does the block optimizer use?**  
A: **OR-Tools CP-SAT** — Constraint Programming with a SAT-based solver from Google. It is NOT ILP/MILP/Genetic Algorithm. CP-SAT handles combinatorial scheduling problems by encoding variables and constraints as Boolean satisfiability problems. CP-SAT se matlab hai constraint programming + SAT solving.

**Q12. What is the optimizer's objective function?**  
A: Maximize the sum of `priority_score × 10,000 × selected` minus a small start-time penalty (favours earlier scheduling). Subject to: each request scheduled at most once; no corridor overlap (AddNoOverlap); department workforce capacity (AddCumulative). Solver: `FEASIBLE` status (not OPTIMAL due to 60-second time limit).

**Q13. What are the three distinct concepts in RailSetu AI?**  
A: (A) ML prediction probability — XGBoost output, 0–100%; (B) Dataset risk_score — pre-computed weighted condition percentile, excluded from model, in cleaned CSV; (C) Operational priority score — rule-based formula: 70% operational_priority + 30% deadline_urgency. These must NOT be conflated.

**Q14. How did you handle class imbalance?**  
A: Multiple strategies: (1) `scale_pos_weight=3.95` in XGBoost (makes positive examples 3.95× heavier); (2) `class_weight="balanced"` in Logistic Regression, Decision Tree, Random Forest, HistGB; (3) stratified train/val/test splits (preserves 20.2% positive rate in all splits); (4) PR-AUC as primary metric (not accuracy).

**Q15. What would you do differently to improve the model?**  
A: (1) Better data generation — ensure target has real causal dependence on features; (2) Feature engineering — ratio features (wear/speed, days_since_maintenance × daily_trips); (3) Anomaly detection framing (one-class classification); (4) Domain expert validation of feature weights; (5) Collect real (non-synthetic) data if feasible.

---

## 15. V5 UPGRADE SUMMARY & ARCHITECTURAL EVOLUTION

1. **Signal Diagnosis Completed**: Mutual information across all 30 features confirmed $< 0.010$. Decision tree depth analysis plateaued at depth 2-3 (ROC-AUC 0.548) and degraded past depth 6, empirically proving that 0.55 is a data-intrinsic ceiling rather than algorithm limitation.
2. **Feature Engineering Ablation**: Formulated 6 interaction features; validation PR-AUC fell from 0.3311 to 0.3283 due to collinear noise. The 30 unengineered features were preserved.
3. **Model Benchmark**: Evaluated 10 distinct models (Dummy, LogReg, DT, RF, ExtraTrees, HistGB, XGBoost, CatBoost, Optuna-tuned LightGBM, Voting Ensemble). XGBoost achieved top Val PR-AUC (0.3320) and Test PR-AUC (0.3296).
4. **Isotonic Calibration**: Trained via 3-fold CV. Dropped held-out test Brier score from 0.2259 to 0.1474.
5. **Feature Selection (1-Std Rule)**: Ablated feature subsets (top 5, 8, 10, 12, 15, 20, 30). Pruning drops PR-AUC monotonically toward 0.202. The full 30 features were preserved under the 1-std stability rule.
6. **Relational Database**: Implemented SQLAlchemy 2 + SQLite (`data/railsetu.db`). Tracks 500 requests, 140 windows, 275 assignments, 225 unscheduled requests, model runs, metrics, threshold tables, training distribution stats, and real-time inference logs with CSV download capability.
7. **Automated Test Suite**: Expanded from 54 to 67 automated tests in pytest with 100% pass rate.

