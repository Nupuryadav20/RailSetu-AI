
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
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =====================================================
# 1. CONFIGURATION
# =====================================================

DATA_DIR = "data/processed/model_ready"
MODEL_DIR = "backend/artifacts/models_v3_interaction"
METRICS_DIR = "backend/artifacts/metrics_v3_interaction"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

RANDOM_STATE = 42

# =====================================================
# 2. LOAD DATA
# =====================================================

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")

y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze().astype(int)
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze().astype(int)

if len(X_train) != len(y_train) or len(X_test) != len(y_test):
    raise ValueError("Feature and target row counts do not match.")

print("Training:", X_train.shape)
print("Test:", X_test.shape)

# =====================================================
# 3. SAME TRAIN / VALIDATION SPLIT
# =====================================================

X_fit_base, X_val_base, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train
)

# =====================================================
# 4. FEATURE ENGINEERING
# =====================================================

def add_interaction_feature(df):
    df = df.copy()

    df["signal_ballast_interaction"] = (
        df["signal_system_status"].fillna("Unknown").astype(str)
        + "_"
        + df["ballast_condition"].fillna("Unknown").astype(str)
    )

    return df


# Baseline: original 30 features
X_fit_base = X_fit_base.copy()
X_val_base = X_val_base.copy()
X_test_base = X_test.copy()

# Interaction: original features + one engineered feature
X_fit_interaction = add_interaction_feature(X_fit_base)
X_val_interaction = add_interaction_feature(X_val_base)
X_test_interaction = add_interaction_feature(X_test_base)

# =====================================================
# 5. SAME XGBOOST CONFIGURATION
# =====================================================

def create_model():
    return XGBClassifier(
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


def create_pipeline(X_fit):
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

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", create_model())
    ])


# =====================================================
# 6. THRESHOLD SELECTION
# =====================================================

def select_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    precision = precision[:-1]
    recall = recall[:-1]

    if len(thresholds) == 0:
        return 0.5

    f2 = (
        5 * precision * recall
    ) / (
        4 * precision + recall + 1e-10
    )

    return float(thresholds[np.argmax(f2)])


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


# =====================================================
# 7. TRAIN BOTH EXPERIMENTS
# =====================================================

experiments = {
    "XGBoost Baseline": (
        X_fit_base, X_val_base, X_test_base
    ),
    "XGBoost Interaction": (
        X_fit_interaction,
        X_val_interaction,
        X_test_interaction
    )
}

pipelines = {}
results = {}
thresholds = {}

print("\nV3 FEATURE INTERACTION EXPERIMENT")
print("=" * 55)

for name, (X_fit, X_val, X_final_test) in experiments.items():

    print("\nTraining:", name)

    pipeline = create_pipeline(X_fit)
    pipeline.fit(X_fit, y_fit)

    val_probabilities = pipeline.predict_proba(X_val)[:, 1]

    threshold = select_threshold(y_val, val_probabilities)

    metrics = calculate_metrics(
        y_val, val_probabilities, threshold
    )

    results[name] = metrics
    thresholds[name] = threshold
    pipelines[name] = pipeline

    print("Validation AP:", round(metrics["average_precision"], 4))
    print("Validation ROC-AUC:", round(metrics["roc_auc"], 4))
    print("Validation Precision:", round(metrics["precision"], 4))
    print("Validation Recall:", round(metrics["recall"], 4))
    print("Selected threshold:", round(threshold, 4))

# =====================================================
# 8. SELECT USING VALIDATION AVERAGE PRECISION
# =====================================================

best_name = max(
    results,
    key=lambda name: results[name]["average_precision"]
)

best_pipeline = pipelines[best_name]
best_threshold = thresholds[best_name]

print("\nSELECTED MODEL:", best_name)

# =====================================================
# 9. FINAL TEST EVALUATION
# =====================================================

X_final_test = experiments[best_name][2]

test_probabilities = best_pipeline.predict_proba(
    X_final_test
)[:, 1]

test_metrics = calculate_metrics(
    y_test, test_probabilities, best_threshold
)

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nFINAL TEST RESULTS")

for key, value in test_metrics.items():
    print(f"{key}: {value}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nClassification Report:")
print(classification_report(
    y_test, test_predictions, zero_division=0
))

# =====================================================
# 10. SAVE ARTIFACTS
# =====================================================

model_path = os.path.join(
    MODEL_DIR, "best_model_v3_interaction.joblib"
)

joblib.dump(best_pipeline, model_path)

comparison = pd.DataFrame([
    {"model": name, **metrics}
    for name, metrics in results.items()
])

comparison_path = os.path.join(
    METRICS_DIR, "validation_comparison.csv"
)

comparison.to_csv(comparison_path, index=False)

summary = {
    "selected_model": best_name,
    "selection_metric": "validation_average_precision",
    "validation_results": results,
    "selected_threshold": best_threshold,
    "final_test_metrics": test_metrics,
    "feature_engineering": "signal_system_status + ballast_condition",
    "baseline_feature_count": int(X_train.shape[1]),
    "interaction_feature_count": int(X_train.shape[1] + 1),
    "test_set_used_for_selection": False,
    "deployment_certification": False
}

summary_path = os.path.join(
    METRICS_DIR, "experiment_summary.json"
)

with open(summary_path, "w") as file:
    json.dump(summary, file, indent=4)

print("\nV3 EXPERIMENT COMPLETED")
print("Selected:", best_name)
print("Model:", model_path)
print("Comparison:", comparison_path)
print("Summary:", summary_path)
print("\nCompare validation AP of both experiments.")
