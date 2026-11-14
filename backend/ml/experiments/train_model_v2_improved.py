
import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score,
    average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = "data/processed/model_ready"
MODEL_DIR = "backend/artifacts/models_v2_improved"
METRICS_DIR = "backend/artifacts/metrics_v2_improved"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

RANDOM_STATE = 42
MIN_PRECISION = 0.30
MIN_RECALL = 0.60

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading prepared dataset...")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze().astype(int)
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze().astype(int)

if X_train.columns.tolist() != X_test.columns.tolist():
    raise ValueError("Training and test feature columns do not match.")

if y_train.nunique() != 2:
    raise ValueError("Expected a binary training target.")

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Training distribution:")
print(y_train.value_counts(normalize=True))

# ============================================================
# 3. FIT / VALIDATION SPLIT
# ============================================================

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train
)

print("\nFit:", X_fit.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# ============================================================
# 4. PREPROCESSING
# ============================================================

numeric_features = X_fit.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_fit.select_dtypes(
    exclude=["number"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),

        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

# ============================================================
# 5. EXPERIMENT CONFIGURATIONS
# ============================================================

negative_count = int((y_fit == 0).sum())
positive_count = int((y_fit == 1).sum())
class_ratio = negative_count / max(positive_count, 1)

experiments = {
    "Logistic Regression Balanced": LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1500,
        random_state=RANDOM_STATE
    ),

    "Random Forest Conservative": RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Random Forest Regularized": RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost Weighted": XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=5,
        reg_alpha=0.1,
        scale_pos_weight=class_ratio,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost Unweighted": XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=5,
        reg_alpha=0.1,
        scale_pos_weight=1,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

# ============================================================
# 6. METRICS
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "f2_score": float(fbeta_score(
            y_true, predictions, beta=2, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "predicted_positive_count": int(predictions.sum())
    }


# ============================================================
# 7. THRESHOLD SELECTION
# ============================================================

def select_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    precision = precision[:-1]
    recall = recall[:-1]

    if len(thresholds) == 0:
        return 0.5, "default_no_thresholds"

    f2 = (
        5 * precision * recall
    ) / (
        4 * precision + recall + 1e-10
    )

    valid = np.where(
        (precision >= MIN_PRECISION) &
        (recall >= MIN_RECALL)
    )[0]

    if len(valid) > 0:
        best_index = valid[np.argmax(f2[valid])]
        return float(thresholds[best_index]), "constraints_satisfied"

    # If constraints are impossible, use the best validation F2.
    best_index = int(np.argmax(f2))
    return float(thresholds[best_index]), "constraints_not_satisfied_best_f2"


# ============================================================
# 8. RUN VALIDATION EXPERIMENTS
# ============================================================

validation_results = []
trained_pipelines = {}
threshold_summary = {}

print("\n======================================")
print("V2 IMPROVED EXPERIMENTS STARTED")
print("======================================")

for name, classifier in experiments.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", clone(classifier))
    ])

    pipeline.fit(X_fit, y_fit)

    probabilities = pipeline.predict_proba(X_val)[:, 1]

    threshold, status = select_threshold(y_val, probabilities)

    metrics = calculate_metrics(
        y_val, probabilities, threshold
    )

    metrics["model"] = name
    metrics["threshold"] = threshold
    metrics["threshold_status"] = status

    validation_results.append(metrics)

    trained_pipelines[name] = pipeline

    threshold_summary[name] = {
        "threshold": threshold,
        "status": status
    }

    print("Validation AP:", round(metrics["average_precision"], 4))
    print("Validation ROC-AUC:", round(metrics["roc_auc"], 4))
    print("Threshold:", round(threshold, 4))
    print("Threshold status:", status)

# ============================================================
# 9. DUMMY BASELINE ON VALIDATION
# ============================================================

dummy = DummyClassifier(strategy="prior")
dummy.fit(X_fit, y_fit)

dummy_val_prob = dummy.predict_proba(X_val)[:, 1]

dummy_val_metrics = calculate_metrics(
    y_val, dummy_val_prob, 0.5
)

dummy_val_metrics["model"] = "Dummy Baseline"
dummy_val_metrics["threshold"] = 0.5
dummy_val_metrics["threshold_status"] = "default"

validation_results.append(dummy_val_metrics)

# ============================================================
# 10. SELECT BEST MODEL USING VALIDATION AP
# ============================================================

comparison = pd.DataFrame(validation_results)

candidate_comparison = comparison[
    comparison["model"] != "Dummy Baseline"
]

best_row = candidate_comparison.loc[
    candidate_comparison["average_precision"].idxmax()
]

best_model_name = best_row["model"]
best_pipeline = trained_pipelines[best_model_name]
best_threshold = threshold_summary[best_model_name]["threshold"]
best_status = threshold_summary[best_model_name]["status"]

print("\n======================================")
print("SELECTED MODEL")
print("======================================")
print("Model:", best_model_name)
print("Validation AP:", round(best_row["average_precision"], 4))
print("Threshold:", round(best_threshold, 4))
print("Threshold status:", best_status)

# ============================================================
# 11. FINAL TEST EVALUATION - SELECTED MODEL ONLY
# ============================================================

print("\nFinal evaluation on untouched test data...")

test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]

test_metrics = calculate_metrics(
    y_test, test_probabilities, best_threshold
)

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nFinal Test Metrics:")
for key, value in test_metrics.items():
    print(f"{key}: {value}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(
    y_test, test_predictions, zero_division=0
))

# Dummy baseline test result for reference.
dummy_test_prob = dummy.predict_proba(X_test)[:, 1]

dummy_test_metrics = calculate_metrics(
    y_test, dummy_test_prob, 0.5
)

# ============================================================
# 12. SAVE SELECTED MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR, "best_model_v2_improved.joblib"
)

joblib.dump(best_pipeline, model_path)

# ============================================================
# 13. SAVE VALIDATION COMPARISON
# ============================================================

comparison_path = os.path.join(
    METRICS_DIR, "validation_model_comparison.csv"
)

comparison.to_csv(comparison_path, index=False)

# ============================================================
# 14. SAVE THRESHOLDS
# ============================================================

threshold_path = os.path.join(
    METRICS_DIR, "model_thresholds.json"
)

with open(threshold_path, "w") as file:
    json.dump(threshold_summary, file, indent=4)

# ============================================================
# 15. SAVE BEST MODEL SUMMARY
# ============================================================

best_info = {
    "best_model": best_model_name,
    "selection_metric": "validation_average_precision",
    "validation_average_precision": float(
        best_row["average_precision"]
    ),
    "threshold": best_threshold,
    "threshold_status": best_status,
    "test_metrics": test_metrics,
    "test_evaluation_policy": "selected_model_only",
    "training_rows": int(len(X_fit)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test))
}

best_path = os.path.join(
    METRICS_DIR, "best_model.json"
)

with open(best_path, "w") as file:
    json.dump(best_info, file, indent=4)

# ============================================================
# 16. SAVE FINAL TEST REPORT
# ============================================================

final_report = {
    "selected_model": best_model_name,
    "threshold": best_threshold,
    "test_metrics": test_metrics,
    "dummy_baseline_test_metrics": dummy_test_metrics,
    "confusion_matrix": confusion_matrix(
        y_test, test_predictions, labels=[0, 1]
    ).tolist()
}

report_path = os.path.join(
    METRICS_DIR, "final_test_metrics.json"
)

with open(report_path, "w") as file:
    json.dump(final_report, file, indent=4)

# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n======================================")
print("V2 IMPROVED TRAINING COMPLETED")
print("======================================")

print("\nValidation comparison:")
print(
    comparison[
        ["model", "average_precision", "roc_auc",
         "precision", "recall", "f1_score", "f2_score"]
    ].sort_values(
        "average_precision", ascending=False
    ).to_string(index=False)
)

print("\nSelected model:", best_model_name)
print("Validation AP:", round(best_row["average_precision"], 4))
print("Final test AP:", round(test_metrics["average_precision"], 4))
print("Dummy test AP:", round(dummy_test_metrics["average_precision"], 4))

print("\nSaved model:", model_path)
print("Saved comparison:", comparison_path)
print("Saved thresholds:", threshold_path)
print("Saved summary:", best_path)
print("Saved final test report:", report_path)

print("\nNOTE:")
print("Model selection used validation Average Precision.")
print("The test set was not used for model or threshold selection.")
print("The saved model is the same fitted model used for validation.")
print("Compare final test AP with the dummy baseline.")
print("These results are experiments, not railway deployment certification.")
