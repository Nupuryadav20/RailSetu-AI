
from functools import lru_cache
from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import HTTPException


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "artifacts"

MODEL_FILES = {
    "XGBoost": "xgboost.joblib",
    "Random Forest": "random_forest.joblib",
    "Logistic Regression": "logistic_regression.joblib",
}


@lru_cache(maxsize=1)
def load_model_bundle():
    metadata_path = ARTIFACTS_DIR / "metrics" / "best_model.json"

    if not metadata_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Best model metadata not found."
        )

    try:
        with open(metadata_path, "r") as file:
            metadata = json.load(file)

        model_name = metadata.get("best_model")
        model_file = MODEL_FILES.get(model_name)

        if not model_file:
            raise ValueError("Unsupported model in metadata.")

        model_path = ARTIFACTS_DIR / "models" / model_file

        if not model_path.exists():
            raise FileNotFoundError(model_file)

        model = joblib.load(model_path)
        threshold = float(metadata.get("threshold", 0.5))

        if not 0 <= threshold <= 1:
            raise ValueError("Invalid model threshold.")

        return model, metadata, threshold

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to load the trained model."
        ) from exc


def predict_records(records):
    if not records:
        raise HTTPException(
            status_code=400,
            detail="At least one input record is required."
        )

    try:
        model, metadata, threshold = load_model_bundle()

        input_df = pd.DataFrame(records)

        required_features = list(model.feature_names_in_)

        missing_features = [
            feature for feature in required_features
            if feature not in input_df.columns
        ]

        if missing_features:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Required features are missing.",
                    "missing_features": missing_features
                }
            )

        input_df = input_df[required_features]

        probabilities = model.predict_proba(input_df)[:, 1]

        predictions = (probabilities >= threshold).astype(int)

        results = []

        for probability, prediction in zip(
            probabilities, predictions
        ):
            results.append({
                "maintenance_probability_percent": round(
                    float(probability) * 100, 2
                ),
                "predicted_class": int(prediction),
                "threshold": threshold
            })

        return {
            "model": metadata["best_model"],
            "selection_metric": metadata.get("selection_metric"),
            "threshold_status": metadata.get("threshold_status"),
            "prediction_type": "Experimental decision-support output",
            "results": results
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be generated."
        ) from exc
