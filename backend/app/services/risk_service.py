
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException

from backend.app.services.model_service import load_model_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_ready" / "X_train.csv"
)

MODEL_WEIGHT = 0.40
CONDITION_WEIGHT = 0.60

CONDITION_FEATURES = {
    "rail_wear_mm": (1.5, "high"),
    "wheel_wear_percent": (1.5, "high"),
    "track_vibration_level": (1.2, "high"),
    "axle_temperature_c": (1.0, "high"),
    "bearing_temperature_c": (1.2, "high"),
    "traction_motor_temp_c": (1.0, "high"),
    "brake_pad_wear_percent": (1.2, "high"),
    "last_maintenance_days": (1.0, "high"),
    "inspection_score": (1.2, "low"),
    "sensor_health_index": (1.0, "low"),
}

CATEGORICAL_FEATURES = {
    "ballast_condition": (["poor"], 1.5),
    "signal_system_status": (["fault"], 1.5),
}


def assign_priority(score):
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def calculate_condition_scores(train_df, input_df):
    weighted_scores = []
    total_weight = 0

    for feature, (weight, direction) in CONDITION_FEATURES.items():
        if feature not in train_df.columns or feature not in input_df.columns:
            continue

        train_values = pd.to_numeric(
            train_df[feature], errors="coerce"
        ).dropna().to_numpy()

        input_values = pd.to_numeric(
            input_df[feature], errors="coerce"
        ).to_numpy()

        if len(train_values) == 0:
            scores = np.full(len(input_df), 50.0)
        else:
            input_values = np.where(
                np.isnan(input_values),
                np.median(train_values),
                input_values,
            )
            scores = (
                np.searchsorted(
                    np.sort(train_values), input_values, side="right"
                ) / len(train_values)
            ) * 100

            if direction == "low":
                scores = 100 - scores

        weighted_scores.append(scores * weight)
        total_weight += weight

    for feature, (concerning_values, weight) in CATEGORICAL_FEATURES.items():
        if feature not in input_df.columns:
            continue

        scores = (
            input_df[feature]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(concerning_values)
            .astype(float)
            .to_numpy() * 100
        )

        weighted_scores.append(scores * weight)
        total_weight += weight

    if total_weight == 0:
        raise ValueError("No condition features available.")

    return np.clip(
        np.sum(weighted_scores, axis=0) / total_weight, 0, 100
    )


def score_records(records):
    if not records:
        raise HTTPException(
            status_code=400,
            detail="At least one input record is required.",
        )

    try:
        model, metadata, _ = load_model_bundle()
        input_df = pd.DataFrame(records)

        required_features = list(model.feature_names_in_)
        missing = [
            feature for feature in required_features
            if feature not in input_df.columns
        ]

        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Required features are missing.",
                    "missing_features": missing,
                },
            )

        input_df = input_df[required_features]
        train_df = pd.read_csv(TRAIN_DATA_PATH)

        probabilities = model.predict_proba(input_df)[:, 1] * 100
        condition_scores = calculate_condition_scores(train_df, input_df)

        priority_scores = np.clip(
            MODEL_WEIGHT * probabilities
            + CONDITION_WEIGHT * condition_scores,
            0,
            100,
        )

        results = []
        for probability, condition, priority in zip(
            probabilities, condition_scores, priority_scores
        ):
            results.append({
                "model_risk_probability_percent": round(float(probability), 2),
                "equipment_condition_score": round(float(condition), 2),
                "operational_priority_score": round(float(priority), 2),
                "priority_level": assign_priority(float(priority)),
            })

        return {
            "model": metadata["best_model"],
            "scoring_type": "Experimental prototype priority score",
            "weights": {
                "model": MODEL_WEIGHT,
                "condition": CONDITION_WEIGHT,
            },
            "results": results,
            "disclaimer": (
                "Not an official railway safety rating or calibrated "
                "failure probability. Requires domain validation."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Risk scoring could not be completed.",
        ) from exc
