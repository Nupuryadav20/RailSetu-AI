
import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = (
    "backend/artifacts/risk_scoring/"
    "maintenance_risk_scores.csv"
)

OUTPUT_DIR = "backend/artifacts/simulation"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "simulated_maintenance_requests.csv"
)

NUMBER_OF_REQUESTS = 500
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# MAINTENANCE TYPES AND DEPARTMENTS
# ============================================================

MAINTENANCE_OPTIONS = {
    "Engineering": [
        "Track Inspection",
        "Rail Wear Repair",
        "Ballast Maintenance",
        "Track Vibration Inspection",
        "Wheel Defect Inspection"
    ],
    "Traction": [
        "Electrical Equipment Inspection",
        "Battery System Maintenance",
        "Traction Motor Inspection",
        "Power Supply Maintenance"
    ],
    "Signal & Telecom": [
        "Signal System Repair",
        "Signal Equipment Inspection",
        "Communication System Maintenance"
    ]
}

DEPARTMENTS = list(MAINTENANCE_OPTIONS.keys())

CORRIDORS = [
    "CORRIDOR-A",
    "CORRIDOR-B",
    "CORRIDOR-C",
    "CORRIDOR-D",
    "CORRIDOR-E"
]


# ============================================================
# LOAD RISK SCORES
# ============================================================

print("\nLoading maintenance risk scores...")

risk_df = pd.read_csv(INPUT_PATH)

if risk_df.empty:
    raise ValueError("Risk scoring file contains no records.")

required_columns = [
    "record_id",
    "model_risk_probability_percent",
    "equipment_condition_score",
    "operational_priority_score",
    "priority_level"
]

missing = [
    col for col in required_columns
    if col not in risk_df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# ============================================================
# SELECT DEMONSTRATION RECORDS
# ============================================================

number_of_requests = min(
    NUMBER_OF_REQUESTS,
    len(risk_df)
)

selected_indices = rng.choice(
    len(risk_df),
    size=number_of_requests,
    replace=False
)

requests = risk_df.iloc[selected_indices].copy()
requests = requests.reset_index(drop=True)

print("Selected records:", len(requests))


# ============================================================
# GENERATE SIMULATED REQUEST DETAILS
# ============================================================

today = pd.Timestamp.now().normalize()

request_ids = []
asset_ids = []
corridor_ids = []
departments = []
maintenance_types = []
locations = []
durations = []
team_sizes = []
earliest_starts = []
deadlines = []

for i in range(len(requests)):

    department = rng.choice(DEPARTMENTS)

    maintenance_type = rng.choice(
        MAINTENANCE_OPTIONS[department]
    )

    # Simulated planning details.
    request_id = f"REQ-{i + 1:04d}"
    asset_id = f"SIM-ASSET-{i + 1:05d}"

    corridor = rng.choice(CORRIDORS)

    location_km = round(
        float(rng.uniform(10, 500)), 2
    )

    duration = int(rng.integers(1, 7))

    team_size = int(rng.integers(2, 9))

    start_day = int(rng.integers(0, 6))

    earliest_start = (
        today
        + pd.Timedelta(days=start_day)
        + pd.Timedelta(hours=8)
    )

    # More urgent requests receive shorter deadlines.
    priority = requests.loc[
        i, "operational_priority_score"
    ]

    if priority >= 70:
        deadline_days = int(rng.integers(1, 3))
    elif priority >= 40:
        deadline_days = int(rng.integers(2, 6))
    else:
        deadline_days = int(rng.integers(4, 8))

    deadline = (
        earliest_start
        + pd.Timedelta(days=deadline_days)
    )

    request_ids.append(request_id)
    asset_ids.append(asset_id)
    corridor_ids.append(corridor)
    departments.append(department)
    maintenance_types.append(maintenance_type)
    locations.append(location_km)
    durations.append(duration)
    team_sizes.append(team_size)
    earliest_starts.append(earliest_start)
    deadlines.append(deadline)


# ============================================================
# BUILD REQUEST DATAFRAME
# ============================================================

requests["request_id"] = request_ids
requests["asset_id"] = asset_ids
requests["corridor_id"] = corridor_ids
requests["department"] = departments
requests["maintenance_type"] = maintenance_types
requests["location_km"] = locations
requests["required_duration_hours"] = durations
requests["team_size_required"] = team_sizes

requests["earliest_start"] = earliest_starts
requests["deadline"] = deadlines

requests["status"] = "Pending"

# Explicitly identify generated planning information.
requests["is_simulated"] = True

# Arrange columns for readability.
output_columns = [
    "request_id",
    "record_id",
    "asset_id",
    "corridor_id",
    "location_km",
    "department",
    "maintenance_type",
    "required_duration_hours",
    "team_size_required",
    "earliest_start",
    "deadline",
    "model_risk_probability_percent",
    "equipment_condition_score",
    "operational_priority_score",
    "priority_level",
    "status",
    "is_simulated"
]

requests = requests[output_columns]

# Save timestamps in ISO-friendly format.
requests["earliest_start"] = pd.to_datetime(
    requests["earliest_start"]
).dt.strftime("%Y-%m-%d %H:%M:%S")

requests["deadline"] = pd.to_datetime(
    requests["deadline"]
).dt.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# SAVE OUTPUT
# ============================================================

requests.to_csv(OUTPUT_PATH, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================================")
print("SIMULATED MAINTENANCE REQUESTS GENERATED")
print("==============================================")

print("\nTotal requests:", len(requests))

print("\nDepartment distribution:")
print(requests["department"].value_counts())

print("\nPriority distribution:")
print(requests["priority_level"].value_counts())

print("\nMaintenance type distribution:")
print(requests["maintenance_type"].value_counts())

print("\nSample requests:")
print(requests.head(5).to_string(index=False))

print("\nSaved file:")
print(OUTPUT_PATH)

print("\nNOTE:")
print("Asset, corridor, location, department, duration,")
print("and scheduling details are simulated demonstration data.")
