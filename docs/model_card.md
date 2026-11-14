# RailSetu AI — Machine Learning Model Card

## 1. Model Overview

| Field | Description |
|---|---|
| Project | RailSetu AI |
| Model Type | Binary Classification |
| Primary Model | XGBoost Classifier |
| Task | Maintenance Requirement Prediction |
| Target Variable | `maintenance_required` |
| Framework | XGBoost, Scikit-learn |
| Model Selection Metric | Validation Average Precision |
| Model Status | Experimental Prototype |

### Purpose

The model estimates whether maintenance is required based on available railway equipment, infrastructure, operational, and environmental features.

It is designed to support maintenance analysis and prioritization in the RailSetu AI prototype.

It is not intended to independently make railway safety or maintenance approval decisions.

## 2. Dataset Information

**Dataset:** Indian Railway Failure Detection and Maintenance 100K

**Source:** Kaggle

The dataset is used for experimental model development. It is synthetic/simulated and is not verified operational Indian Railways data.

### Dataset Statistics

| Attribute | Value |
|---|---:|
| Original records | Approximately 100,300 |
| Duplicate records removed | Approximately 300 |
| Cleaned records | Approximately 100,000 |
| Training split | 80% |
| Testing split | 20% |
| Model input features | 30 |
| Numerical features | 25 |
| Categorical features | 5 |

### Target Distribution

| Target Value | Meaning | Records |
|---|---|---:|
| 0 | Maintenance not required | 79,804 |
| 1 | Maintenance required | 20,196 |

The positive class represents approximately 20.2% of the cleaned dataset.

## 3. Data Preprocessing

The preprocessing workflow included:

1. Loading the original dataset.
2. Inspecting data types and dataset structure.
3. Identifying duplicate records.
4. Removing approximately 300 duplicate records.
5. Checking missing values.
6. Applying median imputation to missing numerical values.
7. Preparing cleaned data for exploratory analysis.
8. Preparing training and testing datasets.

### Excluded Features

The following columns were excluded from model training:

| Feature | Reason |
|---|---|
| `train_id` | Identifier rather than a predictive feature |
| `failure_type` | Direct target leakage risk |
| `failure_severity` | Direct target leakage risk |
| `risk_score` | Excluded from model inputs |

The final prepared dataset contains 30 model input features.

## 4. Model Development

Four classifiers were considered during evaluation.

### Logistic Regression

Used as a baseline classification algorithm.

### Random Forest Classifier

Used to evaluate an ensemble of decision trees and nonlinear relationships.

### XGBoost Classifier

Used as a gradient boosting model and selected based on validation Average Precision.

### Dummy Classifier

Used as a simple baseline for comparison.

## 5. Evaluation Metrics

The following metrics were used:

- Accuracy: Overall proportion of correct predictions.
- Precision: Proportion of positive predictions that are correct.
- Recall: Proportion of actual positive cases identified.
- F1-score: Harmonic mean of precision and recall.
- F2-score: Gives more importance to recall than precision.
- ROC-AUC: Measures ranking ability across classification thresholds.
- Average Precision: Summarizes precision-recall performance.

Average Precision is particularly relevant because the dataset has an imbalanced target distribution.

## 6. Model Evaluation Results

Recorded test-set performance:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6057 | 0.2294 | 0.4038 | 0.2926 | 0.5383 | 0.2529 |
| Random Forest | 0.8134 | 0.7308 | 0.1203 | 0.2066 | 0.5506 | 0.3349 |
| XGBoost | 0.7148 | 0.2779 | 0.2580 | 0.2676 | 0.5508 | 0.3339 |
| Dummy Classifier | 0.7981 | — | 0.0000 | — | — | 0.2020 |

*Results are rounded to four decimal places.*

### Selected Model

XGBoost was selected using validation Average Precision.

Recorded validation Average Precision:

**0.3322**

The model is persisted as a serialized artifact and loaded by the backend model service.

### Threshold

The prediction service currently uses a classification threshold of 0.5.

The recorded threshold-selection process did not produce a feasible alternative under the applied constraints, so the default threshold was retained.

## 7. Explainability

RailSetu AI uses SHAP to analyze model behavior.

Generated explainability outputs include:

- SHAP feature importance.
- SHAP summary plot.
- Individual prediction explanation.
- Individual waterfall visualization.

These outputs help identify which input features contribute to individual and overall model predictions.

SHAP explanations describe model behavior and do not establish causal relationships.

## 8. Model Input and Output

### Input

The prediction service expects a record containing the 30 required model features.

These include:

- Railway region and season.
- Train type and train age.
- Average speed and distance travelled.
- Track temperature and rail wear.
- Track vibration and ballast condition.
- Environmental conditions.
- Wheel and brake-related measurements.
- Bearing and traction motor temperatures.
- Signal system status.
- Power consumption and load factor.
- Daily trips and delay information.
- Last maintenance interval.
- Inspection score.
- Sensor health index.

### Output

The prediction service returns:

- Selected model name.
- Maintenance probability percentage.
- Predicted maintenance class.
- Classification threshold.
- Relevant disclaimer.

The probability is the model's estimated output, not a calibrated real-world probability of railway equipment failure.

## 9. Backend Integration

The selected model is integrated through:

`backend/app/services/model_service.py`

The service:

1. Loads the selected model artifact.
2. Retrieves the model's expected input features.
3. Checks for missing required features.
4. Prepares the input dataframe.
5. Calculates prediction probabilities.
6. Applies the configured classification threshold.
7. Returns prediction results.

The prediction endpoint is:

`POST /api/ml/predict`

## 10. Limitations and Risks

The current model has important limitations.

### Limited Predictive Performance

The recorded ROC-AUC values are approximately 0.54–0.55.

These values indicate limited discrimination between positive and negative classes.

### Synthetic Dataset

The training dataset is synthetic/simulated and does not establish performance on real railway operational data.

### Class Imbalance

The positive maintenance class represents approximately 20.2% of the cleaned dataset.

This affects how accuracy, precision, and recall should be interpreted.

### Generalization

Performance on real railway assets, infrastructure, routes, and operating conditions has not been established.

### Calibration

The output probability has not been demonstrated to be calibrated against real-world failure frequencies.

### Operational Safety

The model must not be used as the sole basis for maintenance approvals, railway block decisions, or safety-critical actions.

## 11. Intended Use

The model is intended for:

- Academic demonstration.
- Research and experimentation.
- Predictive maintenance workflow development.
- API integration demonstrations.
- Maintenance prioritization experiments.
- Future dashboard development.

## 12. Out-of-Scope Use

The model is not approved for:

- Autonomous railway maintenance decisions.
- Railway traffic control.
- Safety certification.
- Official block approval.
- Replacement of qualified engineering inspections.
- Real-world failure probability guarantees.

## 13. Future Improvements

Potential improvements include:

1. Training on verified railway maintenance records.
2. Improving feature quality and feature engineering.
3. Evaluating more classification algorithms.
4. Performing systematic hyperparameter optimization.
5. Evaluating alternative classification thresholds.
6. Calibrating model probabilities.
7. Evaluating performance across railway regions and equipment categories.
8. Monitoring model drift.
9. Introducing model versioning.
10. Establishing an appropriate operational validation process.

## 14. Responsible Use Statement

RailSetu AI is an experimental railway predictive maintenance prototype.

Its machine learning predictions are intended to assist research and demonstrate an analytical workflow.

All real-world railway maintenance, operational, and safety decisions must remain subject to authorized railway procedures, verified data, qualified personnel, and applicable safety requirements.

---

**Model:** XGBoost Classifier
**Task:** Maintenance Requirement Prediction
**Project:** RailSetu AI
**Status:** Experimental Research Prototype
