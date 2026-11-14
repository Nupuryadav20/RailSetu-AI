
import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

DATA_DIR = "data/processed/model_ready"
MODEL_DIR = "backend/artifacts/models"
METRICS_DIR = "backend/artifacts/metrics"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

RANDOM_STATE = 42

# Threshold requirements
MIN_PRECISION = 0.30
MIN_RECALL = 0.60

# If no threshold satisfies both constraints,
# use 0.50 and clearly report that constraints were infeasible.
DEFAULT_THRESHOLD = 0.50


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

print("\nLoading prepared dataset...")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))


# ============================================================
# 3. TRAIN / VALIDATION SPLIT
# ============================================================

# Keep the original test dataset untouched.
# Validation data is used for threshold selection.

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train
)

print("\nFit data shape:", X_fit.shape)
print("Validation data shape:", X_val.shape)
print("Final test data shape:", X_test.shape)


# ============================================================
# 4. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X_fit.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_fit.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ============================================================
# 5. CREATE PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 6. DEFINE MODELS
# ============================================================

negative_count = int((y_fit == 0).sum())
positive_count = int((y_fit == 1).sum())

scale_pos_weight = negative_count / max(positive_count, 1)

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ============================================================
# 7. HELPER FUNCTION: CALCULATE METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "f2_score": float(
            fbeta_score(
                y_true,
                y_pred,
                beta=2,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_prob)
        ),
        "average_precision": float(
            average_precision_score(y_true, y_prob)
        ),
        "predicted_positive_count": int(np.sum(y_pred))
    }


# ============================================================
# 8. HELPER FUNCTION: SELECT THRESHOLD
# ============================================================

def select_threshold(y_true, probabilities):

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    # precision and recall contain one extra final element.
    # Therefore, use [:-1] to align them with thresholds.

    precision_values = precision[:-1]
    recall_values = recall[:-1]

    f2_scores = (
        5 * precision_values * recall_values
    ) / (
        4 * precision_values + recall_values + 1e-10
    )

    valid_indices = np.where(
        (precision_values >= MIN_PRECISION) &
        (recall_values >= MIN_RECALL)
    )[0]

    if len(valid_indices) > 0:

        best_index = valid_indices[
            np.argmax(f2_scores[valid_indices])
        ]

        selected_threshold = float(thresholds[best_index])

        status = "constraints_satisfied"

        print("Threshold selected using precision/recall constraints.")

        print(
            "Validation precision:",
            round(precision_values[best_index], 4)
        )

        print(
            "Validation recall:",
            round(recall_values[best_index], 4)
        )

        print(
            "Validation F2:",
            round(f2_scores[best_index], 4)
        )

    else:

        selected_threshold = DEFAULT_THRESHOLD
        status = "constraints_not_satisfied"

        print("\nWARNING: No validation threshold satisfies both constraints.")
        print("Using the default threshold:", DEFAULT_THRESHOLD)
        print("Do not treat this as a successful constrained optimization.")

    return selected_threshold, status


# ============================================================
# 9. TRAIN AND EVALUATE MODELS
# ============================================================

results = []
threshold_summary = {}
validation_scores = {}

print("\n==============================================")
print("MODEL TRAINING STARTED")
print("==============================================")

for model_name, model in models.items():

    print("\n\nTraining:", model_name)
    print("----------------------------------------------")

    # Create a fresh pipeline for each model.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    # --------------------------------------------------------
    # A. Fit on training subset
    # --------------------------------------------------------

    pipeline.fit(X_fit, y_fit)

    # --------------------------------------------------------
    # B. Predict probabilities on validation data
    # --------------------------------------------------------

    validation_probabilities = pipeline.predict_proba(X_val)[:, 1]

    # --------------------------------------------------------
    # C. Select threshold using validation data only
    # --------------------------------------------------------

    best_threshold, threshold_status = select_threshold(
        y_val,
        validation_probabilities
    )

    threshold_summary[model_name] = {
        "threshold": best_threshold,
        "status": threshold_status
    }

    # Save validation ranking metric for model selection.
    validation_ap = average_precision_score(
        y_val,
        validation_probabilities
    )

    validation_scores[model_name] = float(validation_ap)

    # --------------------------------------------------------
    # D. Refit model on complete training dataset
    # --------------------------------------------------------

    pipeline.fit(X_train, y_train)

    # --------------------------------------------------------
    # E. Evaluate on untouched test dataset
    # --------------------------------------------------------

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        test_probabilities
    )

    test_metrics["model"] = model_name
    test_metrics["threshold"] = best_threshold
    test_metrics["threshold_status"] = threshold_status
    test_metrics["validation_average_precision"] = validation_ap

    results.append(test_metrics)

    # --------------------------------------------------------
    # F. Save trained model
    # --------------------------------------------------------

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    model_path = os.path.join(MODEL_DIR, filename)

    joblib.dump(pipeline, model_path)

    print("\nTest Metrics:")

    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            test_predictions,
            labels=[0, 1]
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    )

    print("Saved model:", model_path)


# ============================================================
# 10. DUMMY BASELINE
# ============================================================

print("\n\n==============================================")
print("DUMMY BASELINE")
print("==============================================")

dummy_model = DummyClassifier(strategy="prior")

dummy_model.fit(X_fit, y_fit)

dummy_probabilities = dummy_model.predict_proba(X_test)[:, 1]

# Default threshold for the baseline.
dummy_predictions = (
    dummy_probabilities >= 0.50
).astype(int)

dummy_metrics = calculate_metrics(
    y_test,
    dummy_predictions,
    dummy_probabilities
)

dummy_metrics["model"] = "Dummy Baseline"
dummy_metrics["threshold"] = 0.50
dummy_metrics["threshold_status"] = "default"
dummy_metrics["validation_average_precision"] = float(
    average_precision_score(
        y_val,
        dummy_model.predict_proba(X_val)[:, 1]
    )
)

results.append(dummy_metrics)

print("\nDummy Baseline Metrics:")

for metric, value in dummy_metrics.items():
    print(f"{metric}: {value}")


# ============================================================
# 11. SAVE MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(results)

comparison_path = os.path.join(
    METRICS_DIR,
    "model_comparison.csv"
)

comparison_df.to_csv(comparison_path, index=False)

print("\n==============================================")
print("MODEL COMPARISON")
print("==============================================")

print(
    comparison_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "f2_score",
            "roc_auc",
            "average_precision",
            "threshold"
        ]
    ].to_string(index=False)
)

print("\nSaved comparison:", comparison_path)


# ============================================================
# 12. SELECT BEST MODEL USING VALIDATION DATA
# ============================================================

# IMPORTANT:
# Do not select the best model using test-set performance.
# Average Precision on validation data is used here.

best_model_name = max(
    validation_scores,
    key=validation_scores.get
)

best_model_info = {
    "best_model": best_model_name,
    "selection_metric": "validation_average_precision",
    "validation_average_precision": validation_scores[best_model_name],
    "threshold": threshold_summary[best_model_name]["threshold"],
    "threshold_status": threshold_summary[best_model_name]["status"]
}

best_model_path = os.path.join(
    METRICS_DIR,
    "best_model.json"
)

with open(best_model_path, "w") as file:
    json.dump(best_model_info, file, indent=4)


# ============================================================
# 13. SAVE THRESHOLD INFORMATION
# ============================================================

threshold_path = os.path.join(
    METRICS_DIR,
    "model_thresholds.json"
)

with open(threshold_path, "w") as file:
    json.dump(threshold_summary, file, indent=4)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("TRAINING COMPLETED")
print("==============================================")

print("\nBest model based on validation Average Precision:")
print(best_model_name)

print(
    "Validation Average Precision:",
    round(validation_scores[best_model_name], 4)
)

print(
    "Selected threshold:",
    threshold_summary[best_model_name]["threshold"]
)

print(
    "Threshold status:",
    threshold_summary[best_model_name]["status"]
)

print("\nSaved files:")
print("- Trained models:", MODEL_DIR)
print("- Model comparison:", comparison_path)
print("- Thresholds:", threshold_path)
print("- Best model summary:", best_model_path)

print("\nIMPORTANT:")
print("Check test ROC-AUC and Average Precision against the Dummy Baseline.")
print("Low scores indicate weak predictive signal in the available features.")
print("Threshold tuning cannot fix a model that cannot distinguish the classes.")
