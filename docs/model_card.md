# RailSetu AI — Machine Learning Model Card (v4)
_Last Updated: 2026-11-14 | System: RailSetu AI Decision Support_

## 1. Model Overview
- **Model Name**: RailSetu Predictive Maintenance Classifier (XGBoost Pipeline v4)
- **Framework**: Scikit-Learn 1.6, XGBoost 2.1
- **Task**: Binary classification (`maintenance_required` ∈ {0, 1})
- **Model Selection Criterion**: Validation Average Precision (PR-AUC)
- **Status**: Research / Academic Prototype (Decision Support Only)

## 2. Intended Use & Boundaries
- **Intended Use**: Assistance in flagging potential rolling stock and track maintenance candidates for operational review.
- **Out of Scope**: Direct automated approval of maintenance actions or safety-critical railway signaling decisions.
- **Data Reality**: Synthetic simulation dataset (100,000 records). Not live Indian Railways telematics data.

## 3. Training & Validation Setup
- **Dataset**: `data/processed/railway_cleaned.csv` (100,000 records × 35 columns)
- **Splits**: 60,000 Train (60%), 20,000 Validation (20%), 20,000 Test (20%) — Stratified Split (Seed 42)
- **Target Distribution**:
  - Class 0 (No maintenance): 79,804 (79.8%)
  - Class 1 (Maintenance required): 20,196 (20.2%)
- **Feature Set (30 Features)**:
  - 25 Numerical features (wear, temperatures, vibration, mileage, speed, environment)
  - 5 Categorical features (`region`, `season`, `train_type`, `ballast_condition`, `signal_system_status`)
- **Explicitly Excluded Columns (Target Leakage Prevention)**:
  - `risk_score` (Derived from target)
  - `failure_type` (Post-event label)
  - `failure_severity` (Post-event severity)
  - `train_id` (Identifier)

## 4. Evaluated Performance Matrix

### Validation Set (Hyperparameter & Threshold Selection)
| Metric | Dummy Baseline | Logistic Regression | Decision Tree | Random Forest | HistGradientBoosting | XGBoost (v4) |
|---|---|---|---|---|---|---|
| **ROC-AUC** | 0.4990 | 0.5321 | 0.5503 | 0.5512 | 0.5521 | **0.5552** |
| **PR-AUC (AP)** | 0.2016 | 0.2509 | 0.3057 | 0.3250 | 0.3260 | **0.3303** |
| **Balanced Acc** | 0.4990 | 0.5183 | 0.5468 | 0.5419 | 0.5476 | **0.5447** |
| **F1 Score** | 0.2002 | 0.3017 | 0.2425 | 0.2402 | 0.2363 | **0.2622** |
| **MCC** | -0.0021 | 0.0294 | 0.1171 | 0.0997 | 0.1263 | **0.0953** |
| **Cohen's Kappa** | -0.0021 | 0.0242 | 0.1112 | 0.0965 | 0.1167 | **0.0948** |
| **Brier Score** | 0.3229 | 0.2484 | 0.2301 | 0.2357 | 0.2294 | **0.2142** |

### Held-out Test Set (Unbiased Final Evaluation)
- **XGBoost (Uncalibrated)**: ROC-AUC: 0.5445 | PR-AUC: 0.3270 | F1: 0.2551 | Balanced Acc: 0.5411 | Precision: 0.2805 | Recall: 0.2340
- **Dummy Classifier Baseline**: ROC-AUC: 0.5039 | PR-AUC: 0.2033 | F1: 0.2081

## 5. Threshold Operating Analysis
- **Threshold 0.50 (Default fallback)**: Precision = 28.6%, Recall = 24.2% (Flags 17.1% of items).
- **Threshold 0.45 (Balanced)**: Precision = 22.5%, Recall = 50.1% (Flags 44.9% of items).
- **Threshold 0.05-0.20 (High recall)**: Recall ≈ 100%, but precision approaches base rate (20.2%), flagging almost entire fleet.

## 6. Explainability & Feature Attributions
- Top predictive signals identified via SHAP: `rail_wear_mm`, `track_vibration_level`, `wheel_wear_percent`, `bearing_temperature_c`.
- Permutation importance confirms `ballast_condition`, `signal_system_status`, and `rail_wear_mm` yield top loss drops when permuted.

## 7. Operational Separation of Three Concepts
1. **Concept A (ML Prediction Probability)**: Statistical likelihood ($0-100\%$) that maintenance is needed, computed by the supervised ML model from 30 telemetry features.
2. **Concept B (Dataset Risk Score)**: Synthetic composite label present in dataset column `risk_score` (strictly excluded during model training to avoid leakage).
3. **Concept C (Operational Priority Score)**: Deterministic business-rule priority score ($0-100$) produced by `priority_engine.py` combining model probabilities with equipment wear thresholds, corridor importance, and safety criteria for CP-SAT schedule optimization.

---

## 8. V5 Model Upgrade (Active Serving Pipeline)

### Model Identity
- **Model Name**: RailSetu V5 Calibrated XGBoost Pipeline
- **Active Serving File**: `backend/artifacts/models/v5/best_model_v5.joblib` + `preprocessor_v5.joblib` + `best_model_v5_calibrated.joblib`
- **Metadata**: `backend/artifacts/models/v5/v5_metadata.json`
- **MLflow Tracking**: Experiment `railsetu_v5`

### Quantitative Performance
- **Validation PR-AUC**: **0.3320** vs V4 0.3303 (margin: +0.0018, within CV std 0.0055)
- **Held-Out Test PR-AUC**: **0.3296** vs V4 0.3270
- **Held-Out Test ROC-AUC**: **0.5451** vs V4 0.5445
- **Calibration (Isotonic Regression)**: Reduced held-out test Brier score from **0.2259** to **0.1474**
- **Operating Threshold**: **0.45** (flags ~15% fleet, precision 23.5%, recall 17.5%, F1 0.201)
- **Feature Selection (Task 2)**: 30 evaluated features retained based on empirical 1-std boundary rule (feature pruning dropped PR-AUC toward the 0.202 dummy baseline).

