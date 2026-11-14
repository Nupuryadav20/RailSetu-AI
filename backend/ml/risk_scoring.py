
import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = "backend/artifacts/models/xgboost.joblib"

TRAIN_DATA_PATH = "data/processed/model_ready/X_train.csv"
TEST_DATA_PATH = "data/processed/model_ready/X_test.csv"

OUTPUT_DIR = "backend/artifacts/risk_scoring"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prototype weights, not official railway standards.
MODEL_WEIGHT = 0.40
CONDITION_WEIGHT = 0.60

# ============================================================
# 2. CONDITION INDICATORS AND WEIGHTS
# ============================================================

# Higher values indicate a more concerning condition.
# These are prototype assumptions to be validated with
# domain experts before any operational deployment.

CONDITION_FEATURES = {
    "rail_wear_mm": {
        "weight": 1.5,
        "direction": "high"
    },
    "wheel_wear_percent": {
        "weight": 1.5,
        "direction": "high"
    },
    "track_vibration_level": {
        "weight": 1.2,
        "direction": "high"
    },
    "axle_temperature_c": {
        "weight": 1.0,
        "direction": "high"
    },
    "bearing_temperature_c": {
        "weight": 1.2,
        "direction": "high"
    },
    "traction_motor_temp_c": {
        "weight": 1.0,
        "direction": "high"
    },
    "brake_pad_wear_percent": {
        "weight": 1.2,
        "direction": "high"
    },
    "last_maintenance_days": {
        "weight": 1.0,
        "direction": "high"
    },
    "inspection_score": {
        "weight": 1.2,
        "direction": "low"
    },
    "sensor_health_index": {
        "weight": 1.0,
        "direction": "low"
    }
}

# Categorical warning indicators.
CATEGORICAL_FEATURES = {
    "ballast_condition": {
        "concerning_values": ["poor"],
        "weight": 1.5
    },
    "signal_system_status": {
        "concerning_values": ["fault"],
        "weight": 1.5
    }
}


# ============================================================
# 3. LOAD MODEL AND DATA
# ============================================================

print("\nLoading trained XGBoost model...")

model = joblib.load(MODEL_PATH)

X_train = pd.read_csv(TRAIN_DATA_PATH)
X_test = pd.read_csv(TEST_DATA_PATH)

print("Training dataset:", X_train.shape)
print("Test dataset:", X_test.shape)


# ============================================================
# 4. CALCULATE RELATIVE NUMERIC CONDITION SCORES
# ============================================================

def calculate_numeric_score(train_series, test_series, direction):

    train_values = pd.to_numeric(
        train_series,
        errors="coerce"
    ).dropna().to_numpy()

    test_values = pd.to_numeric(
        test_series,
        errors="coerce"
    ).to_numpy()

    if len(train_values) == 0:
        return np.full(len(test_values), 50.0)

    # Replace missing test values with training median.
    median_value = np.median(train_values)

    test_values = np.where(
        np.isnan(test_values),
        median_value,
        test_values
    )

    sorted_train = np.sort(train_values)

    # Calculate relative percentile against training data.
    percentile_scores = (
        np.searchsorted(
            sorted_train,
            test_values,
            side="right"
        ) / len(sorted_train)
    ) * 100

    if direction == "low":
        percentile_scores = 100 - percentile_scores

    return np.clip(percentile_scores, 0, 100)


# ============================================================
# 5. CALCULATE EQUIPMENT CONDITION SCORE
# ============================================================

def calculate_condition_scores(train_df, test_df):

    weighted_scores = []
    total_weight = 0

    print("\nCalculating equipment condition scores...")

    for feature, config in CONDITION_FEATURES.items():

        if feature not in train_df.columns:
            print("Skipping missing training feature:", feature)
            continue

        if feature not in test_df.columns:
            print("Skipping missing test feature:", feature)
            continue

        feature_scores = calculate_numeric_score(
            train_df[feature],
            test_df[feature],
            config["direction"]
        )

        weighted_scores.append(
            feature_scores * config["weight"]
        )

        total_weight += config["weight"]

    # Categorical warning indicators.
    for feature, config in CATEGORICAL_FEATURES.items():

        if feature not in train_df.columns or feature not in test_df.columns:
            print("Skipping missing categorical feature:", feature)
            continue

        concerning_values = [
            value.lower()
            for value in config["concerning_values"]
        ]

        feature_scores = (
            test_df[feature]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(concerning_values)
            .astype(float)
            .to_numpy() * 100
        )

        weighted_scores.append(
            feature_scores * config["weight"]
        )

        total_weight += config["weight"]

    if total_weight == 0:
        raise ValueError("No valid condition features found.")

    condition_score = (
        np.sum(weighted_scores, axis=0) / total_weight
    )

    return np.clip(condition_score, 0, 100)


# ============================================================
# 6. GENERATE MODEL PROBABILITIES
# ============================================================

print("\nGenerating XGBoost predictions...")

model_probabilities = model.predict_proba(X_test)[:, 1]

# Express probability as a percentage.
model_risk_percentage = model_probabilities * 100


# ============================================================
# 7. GENERATE CONDITION SCORES
# ============================================================

condition_scores = calculate_condition_scores(
    X_train,
    X_test
)


# ============================================================
# 8. CALCULATE OPERATIONAL PRIORITY SCORE
# ============================================================

# This is a prototype planning score.
# It is NOT a calibrated failure probability.

priority_scores = (
    MODEL_WEIGHT * model_risk_percentage
    + CONDITION_WEIGHT * condition_scores
)

priority_scores = np.clip(priority_scores, 0, 100)


# ============================================================
# 9. ASSIGN PROTOTYPE PRIORITY BANDS
# ============================================================

def assign_priority(score):

    if score >= 70:
        return "High"

    elif score >= 40:
        return "Medium"

    else:
        return "Low"


priority_bands = [
    assign_priority(score)
    for score in priority_scores
]


# ============================================================
# 10. CREATE FINAL OUTPUT
# ============================================================

results = pd.DataFrame({
    "record_id": np.arange(1, len(X_test) + 1),
    "model_risk_probability_percent": np.round(
        model_risk_percentage, 2
    ),
    "equipment_condition_score": np.round(
        condition_scores, 2
    ),
    "operational_priority_score": np.round(
        priority_scores, 2
    ),
    "priority_level": priority_bands
})


# ============================================================
# 11. SAVE RESULTS
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "maintenance_risk_scores.csv"
)

results.to_csv(output_path, index=False)


# ============================================================
# 12. DISPLAY SUMMARY
# ============================================================

print("\n==============================================")
print("MAINTENANCE RISK SCORING COMPLETED")
print("==============================================")

print("\nPriority distribution:")

print(
    results["priority_level"]
    .value_counts()
    .reindex(["High", "Medium", "Low"], fill_value=0)
)

print("\nAverage scores:")

print(
    results[
        [
            "model_risk_probability_percent",
            "equipment_condition_score",
            "operational_priority_score"
        ]
    ].mean().round(2)
)

print("\nTop 10 priority records:")

print(
    results.sort_values(
        "operational_priority_score",
        ascending=False
    ).head(10).to_string(index=False)
)

print("\nSaved output:", output_path)

print("\nIMPORTANT:")
print("Model probability and condition score are separate measures.")
print("Priority score is a configurable prototype ranking.")
print("It is not an official railway safety rating.")
