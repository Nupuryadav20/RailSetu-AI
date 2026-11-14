from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
import json
import math
import io
import csv

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from backend.app.database.connection import SessionLocal
from backend.app.database.models import Prediction, FeatureStat

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"
MODELS_V5 = ARTIFACTS_DIR / "models" / "v5"
MODELS_V4 = ARTIFACTS_DIR / "models" / "v4"

MODEL_FILES = {
    "XGBoost": "xgboost.joblib",
    "Random Forest": "random_forest.joblib",
    "Logistic Regression": "logistic_regression.joblib",
}


def _norm_cdf(x):
    """Standard normal cumulative distribution function."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _try_load_v5():
    """Try loading v5 pipeline + metadata. Returns (pipe, meta, threshold, 'v5') or None."""
    meta_path = MODELS_V5 / "v5_metadata.json"
    model_path = MODELS_V5 / "best_model_v5.joblib"
    prep_path = MODELS_V5 / "preprocessor_v5.joblib"
    cal_path = MODELS_V5 / "best_model_v5_calibrated.joblib"

    if not (meta_path.exists() and model_path.exists()):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    if not meta.get("switch_api_to_v5", False):
        return None

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path) if prep_path.exists() else None
    calibrated = joblib.load(cal_path) if cal_path.exists() else None

    if preprocessor is not None:
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        if calibrated is not None:
            pipe.calibrated_pipeline = Pipeline([("preprocessor", preprocessor), ("model", calibrated)])
    else:
        pipe = model

    threshold = float(meta.get("chosen_threshold", 0.45))
    return pipe, meta, threshold, "v5"


def _try_load_v4():
    """Try loading v4 model + metadata. Returns (model, meta, threshold, 'v4') or None."""
    meta_path = MODELS_V4 / "v4_metadata.json"
    model_path = MODELS_V4 / "best_model_v4.joblib"
    if not (meta_path.exists() and model_path.exists()):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    if not meta.get("switch_api_to_v4", False):
        return None
    model = joblib.load(model_path)
    threshold = float(meta.get("chosen_threshold", 0.5))
    return model, meta, threshold, "v4"


@lru_cache(maxsize=1)
def load_model_bundle():
    """Load v5 if configured, else v4, else v1. Returns (model, metadata, threshold, version)."""
    v5 = _try_load_v5()
    if v5 is not None:
        return v5

    v4 = _try_load_v4()
    if v4 is not None:
        return v4

    # v1 fallback
    metadata_path = ARTIFACTS_DIR / "metrics" / "best_model.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=500, detail="Best model metadata not found.")
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        model_name = metadata.get("best_model")
        model_file = MODEL_FILES.get(model_name)
        if not model_file:
            raise ValueError("Unsupported model in metadata.")
        model_path = ARTIFACTS_DIR / "models" / model_file
        if not model_path.exists():
            raise FileNotFoundError(model_file)
        model = joblib.load(model_path)
        threshold = float(metadata.get("threshold", 0.5))
        return model, metadata, threshold, "v1"
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to load the trained model.") from exc


@lru_cache(maxsize=1)
def get_feature_stats_map():
    """Load feature distribution statistics from DB or fallback."""
    stats = {}
    db: Session = SessionLocal()
    try:
        records = db.query(FeatureStat).all()
        for r in records:
            stats[r.feature] = {
                "data_type": r.data_type,
                "min": r.min_val,
                "max": r.max_val,
                "mean": r.mean_val,
                "std": r.std_val,
                "p25": r.p25,
                "p50": r.p50,
                "p75": r.p75,
                "class_0_mean": r.class_0_mean,
                "class_1_mean": r.class_1_mean,
            }
    except Exception:
        pass
    finally:
        db.close()
    return stats


def _compute_shap(model, input_df, feature_cols):
    """Compute SHAP values for single record using TreeExplainer or fallback."""
    try:
        import shap
        if hasattr(model, "named_steps"):
            preprocessor = model.named_steps.get("preprocessor")
            estimator = model.named_steps.get("model")
            X = preprocessor.transform(input_df) if preprocessor else input_df.values
        elif hasattr(model, "__getitem__"):
            preprocessor = model["preprocessor"] if "preprocessor" in model.named_steps else None
            estimator = model["model"]
            X = preprocessor.transform(input_df) if preprocessor else input_df.values
        else:
            estimator = model
            X = input_df.values

        explainer = shap.TreeExplainer(estimator)
        sv = explainer(X).values[0]
        result = {}
        for f, v in zip(feature_cols, sv):
            result[f] = round(float(v), 5)
        return result
    except Exception:
        return {}


def predict_records(records):
    if not records:
        raise HTTPException(status_code=400, detail="At least one input record is required.")

    try:
        model, metadata, threshold, version = load_model_bundle()
        input_df = pd.DataFrame(records)

        # Feature requirements
        if version == "v5" and "selected_features" in metadata:
            required_features = metadata["selected_features"]
        elif version == "v4" and "feature_cols" in metadata:
            required_features = metadata["feature_cols"]
        elif hasattr(model, "feature_names_in_"):
            required_features = list(model.feature_names_in_)
        else:
            required_features = list(input_df.columns)

        missing_features = [f for f in required_features if f not in input_df.columns]
        if missing_features:
            raise HTTPException(
                status_code=422,
                detail={"message": "Required features are missing.", "missing_features": missing_features},
            )

        input_df = input_df[required_features]

        # Raw and calibrated inference
        raw_probs = model.predict_proba(input_df)[:, 1]
        cal_probs = raw_probs
        if hasattr(model, "calibrated_pipeline") and model.calibrated_pipeline is not None:
            try:
                cal_probs = model.calibrated_pipeline.predict_proba(input_df)[:, 1]
            except Exception:
                cal_probs = raw_probs

        predictions = (raw_probs >= threshold).astype(int)

        # SHAP calculation for single record
        shap_dict = {}
        if len(records) == 1:
            shap_dict = _compute_shap(model, input_df, required_features)

        stats_map = get_feature_stats_map()
        timestamp_now = datetime.now(timezone.utc).isoformat()

        results = []
        db: Session = SessionLocal()

        try:
            for idx, (raw_p, cal_p, pred) in enumerate(zip(raw_probs, cal_probs, predictions)):
                row_dict = records[idx]
                row_warnings = []
                diagnostics = []

                # Build per-feature diagnostic report
                for f in required_features:
                    val = row_dict.get(f)
                    f_stat = stats_map.get(f, {})
                    z_score = None
                    pct_rank = None
                    c0_m = f_stat.get("class_0_mean")
                    c1_m = f_stat.get("class_1_mean")
                    mean_val = f_stat.get("mean")
                    std_val = f_stat.get("std")

                    if f_stat.get("data_type") == "numeric" and val is not None:
                        try:
                            f_val = float(val)
                            if std_val and std_val > 0:
                                z_score = round((f_val - mean_val) / std_val, 2)
                                pct_rank = round(_norm_cdf(z_score) * 100, 1)

                            min_v = f_stat.get("min")
                            max_v = f_stat.get("max")
                            if min_v is not None and f_val < min_v:
                                row_warnings.append(f"{f} ({f_val}) is below training min ({min_v})")
                            elif max_v is not None and f_val > max_v:
                                row_warnings.append(f"{f} ({f_val}) exceeds training max ({max_v})")
                            elif z_score is not None and abs(z_score) > 3.0:
                                row_warnings.append(f"{f} has unusual z-score ({z_score})")
                        except (ValueError, TypeError):
                            pass

                    diagnostics.append({
                        "feature": f,
                        "value": val,
                        "train_mean": mean_val,
                        "train_std": std_val,
                        "z_score": z_score,
                        "percentile_rank": pct_rank,
                        "class_0_mean": c0_m,
                        "class_1_mean": c1_m,
                        "shap_contribution": shap_dict.get(f, 0.0),
                    })

                # Sort top SHAP contributors
                top_shap = sorted(
                    [{"feature": f, "shap_value": v} for f, v in shap_dict.items()],
                    key=lambda x: abs(x["shap_value"]),
                    reverse=True,
                )[:5]

                result_item = {
                    "maintenance_probability_percent": round(float(raw_p) * 100, 2),
                    "calibrated_probability_percent": round(float(cal_p) * 100, 2),
                    "predicted_class": int(pred),
                    "threshold": threshold,
                    "decision": "MAINTENANCE REQUIRED" if pred == 1 else "NO MAINTENANCE REQUIRED",
                    "diagnostics": diagnostics,
                    "warnings": row_warnings,
                    "top_shap_contributors": top_shap,
                }
                results.append(result_item)

                # Persist to database
                db_record = Prediction(
                    timestamp=timestamp_now,
                    model_name=metadata.get("best_model", "XGBoost"),
                    model_version=version,
                    input_json=json.dumps(row_dict),
                    probability=round(float(raw_p), 4),
                    threshold=threshold,
                    predicted_class=int(pred),
                    top_shap_json=json.dumps(top_shap),
                    warnings_json=json.dumps(row_warnings),
                )
                db.add(db_record)

            db.commit()
        finally:
            db.close()

        response = {
            "model": metadata.get("best_model", "XGBoost"),
            "model_version": version,
            "is_calibrated": hasattr(model, "calibrated_pipeline") and model.calibrated_pipeline is not None,
            "timestamp": timestamp_now,
            "threshold": threshold,
            "threshold_basis": metadata.get("threshold_basis", "balanced_capacity_screening_0.45"),
            "prediction_type": "Experimental decision-support output — synthetic data only",
            "honest_note": (
                "Model ROC-AUC ~0.55 (barely above random 0.50). "
                "Trained on synthetic data. NOT a certified safety system."
            ),
            "concept_note": (
                "This is Concept A: ML probability. Visibly separate from "
                "Concept B (dataset risk_score) and Concept C (rule-based operational priority)."
            ),
            "results": results,
        }
        return response

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction could not be generated.") from exc


def get_prediction_history(
    limit: int = 50,
    model_version: str | None = None,
    predicted_class: int | None = None,
    min_prob: float | None = None,
    max_prob: float | None = None,
):
    """Retrieve filtered predictions from SQLite database."""
    db: Session = SessionLocal()
    try:
        query = db.query(Prediction)
        if model_version:
            query = query.filter(Prediction.model_version == model_version)
        if predicted_class is not None:
            query = query.filter(Prediction.predicted_class == predicted_class)
        if min_prob is not None:
            query = query.filter(Prediction.probability >= min_prob)
        if max_prob is not None:
            query = query.filter(Prediction.probability <= max_prob)

        records = query.order_by(Prediction.id.desc()).limit(limit).all()
        out = []
        for r in records:
            out.append({
                "id": r.id,
                "timestamp": r.timestamp,
                "model_name": r.model_name,
                "model_version": r.model_version,
                "probability": r.probability,
                "probability_percent": round(r.probability * 100, 2) if r.probability is not None else None,
                "threshold": r.threshold,
                "predicted_class": r.predicted_class,
                "top_shap": json.loads(r.top_shap_json) if r.top_shap_json else [],
                "warnings": json.loads(r.warnings_json) if r.warnings_json else [],
            })
        return out
    finally:
        db.close()


def export_predictions_csv(limit: int = 500, model_version: str | None = None) -> str:
    """Generate CSV string of saved predictions."""
    history = get_prediction_history(limit=limit, model_version=model_version)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "timestamp", "model_name", "model_version", "probability", "threshold", "predicted_class"])
    for h in history:
        writer.writerow([
            h["id"],
            h["timestamp"],
            h["model_name"],
            h["model_version"],
            h["probability"],
            h["threshold"],
            h["predicted_class"],
        ])
    return output.getvalue()
