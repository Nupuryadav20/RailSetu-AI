"""
P3 API: Updated model_service.py with v4 support, SHAP contributions,
        model version/timestamp in predict response, prediction history.
"""
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"
REPORTS_V4    = PROJECT_ROOT / "reports" / "v4"
MODELS_V4     = ARTIFACTS_DIR / "models" / "v4"

# Simple in-memory prediction history (last 50; no sensitive data)
_prediction_history: deque = deque(maxlen=50)

# ─── MODEL LOADING ───────────────────────────────────────────────────────────

def _load_v4_bundle():
    """Try to load v4 model and metadata."""
    meta_path = MODELS_V4 / "v4_metadata.json"
    model_path = MODELS_V4 / "best_model_v4.joblib"
    if not meta_path.exists() or not model_path.exists():
        return None, None, None
    with open(meta_path) as f:
        meta = json.load(f)
    model = joblib.load(model_path)
    threshold = float(meta.get("chosen_threshold", 0.5))
    return model, meta, threshold


def _load_v1_bundle():
    """Fallback: v1 XGBoost from artifacts/models/."""
    MODEL_FILES = {
        "XGBoost": "xgboost.joblib",
        "Random Forest": "random_forest.joblib",
        "Logistic Regression": "logistic_regression.joblib",
    }
    meta_path = ARTIFACTS_DIR / "metrics" / "best_model.json"
    if not meta_path.exists():
        raise HTTPException(status_code=500, detail="No model metadata found.")
    with open(meta_path) as f:
        meta = json.load(f)
    model_name = meta.get("best_model")
    model_file = MODEL_FILES.get(model_name)
    if not model_file:
        raise HTTPException(status_code=500, detail="Unsupported model in metadata.")
    model_path = ARTIFACTS_DIR / "models" / model_file
    if not model_path.exists():
        raise HTTPException(status_code=500, detail=f"Model file not found: {model_file}")
    model = joblib.load(model_path)
    threshold = float(meta.get("threshold", 0.5))
    return model, meta, threshold


@lru_cache(maxsize=1)
def load_model_bundle():
    """Load v4 if available, else fall back to v1. Cached."""
    try:
        model, meta, threshold = _load_v4_bundle()
        if model is not None:
            return model, meta, threshold, "v4"
    except Exception:
        pass
    model, meta, threshold = _load_v1_bundle()
    return model, meta, threshold, "v1"


def _get_shap_top_contributors(model, input_df, feature_names, n=5):
    """Return top-N SHAP contributors for a single prediction. Fails gracefully."""
    try:
        import shap
        # Try XGBoost direct explainer on the model's final step
        estimator = model["model"] if hasattr(model, "__getitem__") else model
        preprocessor = model["preprocessor"] if hasattr(model, "__getitem__") else None

        if preprocessor is not None:
            X_transformed = preprocessor.transform(input_df)
        else:
            X_transformed = input_df.values

        explainer = shap.Explainer(estimator)
        shap_vals = explainer(X_transformed)
        sv = shap_vals.values[0] if len(shap_vals.values.shape) == 2 else shap_vals.values[0, :, 1]
        top = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)[:n]
        return [{"feature": f, "shap_value": round(float(v), 5)} for f, v in top]
    except Exception:
        return []


# ─── PREDICTION ──────────────────────────────────────────────────────────────

def predict_records(records):
    if not records:
        raise HTTPException(status_code=400, detail="At least one input record is required.")

    try:
        model, metadata, threshold, version = load_model_bundle()
        input_df = pd.DataFrame(records)

        # Determine feature names
        if version == "v4":
            required_features = metadata.get("feature_cols", [])
        else:
            required_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else []

        missing = [f for f in required_features if f not in input_df.columns]
        if missing:
            raise HTTPException(
                status_code=422,
                detail={"message": "Required features missing.", "missing_features": missing}
            )

        input_df = input_df[required_features]
        probabilities = model.predict_proba(input_df)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        # SHAP for single record (best-effort)
        shap_contribs = []
        if len(records) == 1:
            shap_contribs = _get_shap_top_contributors(model, input_df, required_features, n=5)

        results = []
        for prob, pred in zip(probabilities, predictions):
            results.append({
                "maintenance_probability_percent": round(float(prob) * 100, 2),
                "predicted_class": int(pred),
                "threshold": threshold,
                "decision": "MAINTENANCE REQUIRED" if pred == 1 else "NO MAINTENANCE REQUIRED",
            })

        response = {
            "model": metadata.get("best_model", "unknown"),
            "model_version": version,
            "is_calibrated": metadata.get("is_calibrated", False),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "selection_metric": metadata.get("selection_metric", "validation_average_precision"),
            "threshold": threshold,
            "threshold_basis": metadata.get("threshold_basis", "default_0.5"),
            "prediction_type": "Experimental decision-support output — synthetic data only",
            "honest_note": (
                "Model ROC-AUC ~0.55 (barely above random). "
                "Synthetic dataset. NOT a certified safety system."
            ),
            "concept_note": (
                "This probability (Concept A) is the ML model output. "
                "Separate from dataset risk_score (Concept B) and "
                "operational priority score (Concept C, rule-based)."
            ),
            "shap_top_contributors": shap_contribs,
            "results": results,
        }

        # Append to history (safe — no raw input data stored)
        _prediction_history.append({
            "timestamp": response["timestamp"],
            "model_version": version,
            "threshold": threshold,
            "predicted_class": results[0]["predicted_class"] if results else None,
            "probability_percent": results[0]["maintenance_probability_percent"] if results else None,
        })

        return response

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction could not be generated.") from exc


def get_prediction_history():
    """Return last 50 prediction metadata (no raw input)."""
    return list(_prediction_history)
