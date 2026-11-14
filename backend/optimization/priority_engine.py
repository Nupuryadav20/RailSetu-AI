
import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = (
    "backend/artifacts/simulation/"
    "simulated_maintenance_requests.csv"
)

OUTPUT_DIR = "backend/artifacts/priority"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "ranked_maintenance_requests.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prototype weights. These are configurable assumptions.
OPERATIONAL_WEIGHT = 0.70
DEADLINE_WEIGHT = 0.30


# ============================================================
# VALIDATION
# ============================================================

def load_requests():

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "request_id",
        "department",
        "maintenance_type",
        "operational_priority_score",
        "earliest_start",
        "deadline",
        "required_duration_hours",
        "status"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("No maintenance requests found.")

    return df


# ============================================================
# DEADLINE URGENCY CALCULATION
# ============================================================

def calculate_deadline_urgency(deadline, current_time):

    hours_remaining = (
        deadline - current_time
    ).total_seconds() / 3600

    if hours_remaining <= 0:
        return 100.0

    days_remaining = hours_remaining / 24

    if days_remaining <= 1:
        return 95.0

    elif days_remaining <= 2:
        return 85.0

    elif days_remaining <= 3:
        return 75.0

    elif days_remaining <= 5:
        return 60.0

    elif days_remaining <= 7:
        return 40.0

    else:
        return 20.0


# ============================================================
# FINAL PRIORITY CLASSIFICATION
# ============================================================

def assign_priority(score):

    if score >= 85:
        return "Critical"

    elif score >= 70:
        return "High"

    elif score >= 40:
        return "Medium"

    else:
        return "Low"


# ============================================================
# PRIORITY ENGINE
# ============================================================

def generate_priority_ranking(df):

    print("\nCalculating maintenance priorities...")

    current_time = pd.Timestamp.now()

    df["earliest_start"] = pd.to_datetime(
        df["earliest_start"],
        errors="coerce"
    )

    df["deadline"] = pd.to_datetime(
        df["deadline"],
        errors="coerce"
    )

    if df["deadline"].isna().any():
        raise ValueError(
            "Some maintenance requests have invalid deadlines."
        )

    # Validate operational scores.
    df["operational_priority_score"] = pd.to_numeric(
        df["operational_priority_score"],
        errors="coerce"
    )

    if df["operational_priority_score"].isna().any():
        raise ValueError(
            "Invalid operational priority scores found."
        )

    df["operational_priority_score"] = (
        df["operational_priority_score"].clip(0, 100)
    )

    # Calculate deadline urgency.
    df["deadline_urgency_score"] = df["deadline"].apply(
        lambda deadline: calculate_deadline_urgency(
            deadline,
            current_time
        )
    )

    # Combine operational priority and deadline urgency.
    df["final_priority_score"] = (
        OPERATIONAL_WEIGHT
        * df["operational_priority_score"]
        +
        DEADLINE_WEIGHT
        * df["deadline_urgency_score"]
    )

    df["final_priority_score"] = (
        df["final_priority_score"].clip(0, 100).round(2)
    )

    # Assign priority category.
    df["final_priority_level"] = (
        df["final_priority_score"].apply(assign_priority)
    )

    # Calculate time remaining until deadline.
    df["hours_until_deadline"] = (
        (
            df["deadline"] - current_time
        ).dt.total_seconds() / 3600
    ).round(2)

    df["deadline_status"] = np.where(
        df["hours_until_deadline"] < 0,
        "Overdue",
        "Within Deadline"
    )

    # Rank all requests.
    df = df.sort_values(
        by=[
            "final_priority_score",
            "hours_until_deadline"
        ],
        ascending=[False, True]
    ).reset_index(drop=True)

    df["priority_rank"] = np.arange(
        1,
        len(df) + 1
    )

    return df


# ============================================================
# DEPARTMENT SUMMARY
# ============================================================

def create_department_summary(df):

    summary = (
        df.groupby("department")
        .agg(
            total_requests=("request_id", "count"),
            average_priority=("final_priority_score", "mean"),
            critical_requests=(
                "final_priority_level",
                lambda x: (x == "Critical").sum()
            ),
            high_requests=(
                "final_priority_level",
                lambda x: (x == "High").sum()
            ),
            overdue_requests=(
                "deadline_status",
                lambda x: (x == "Overdue").sum()
            )
        )
        .reset_index()
    )

    summary["average_priority"] = (
        summary["average_priority"].round(2)
    )

    return summary


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    print("\n==============================================")
    print("RAILSETU AI - MAINTENANCE PRIORITY ENGINE")
    print("==============================================")

    print("\nLoading maintenance requests...")

    df = load_requests()

    print("Total requests:", len(df))

    # Rank requests.
    ranked_df = generate_priority_ranking(df)

    # Save ranked requests.
    ranked_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # Department summary.
    department_summary = create_department_summary(
        ranked_df
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "department_priority_summary.csv"
    )

    department_summary.to_csv(
        summary_path,
        index=False
    )

    # Display priority distribution.
    print("\n==============================================")
    print("PRIORITY DISTRIBUTION")
    print("==============================================")

    print(
        ranked_df["final_priority_level"]
        .value_counts()
        .reindex(
            ["Critical", "High", "Medium", "Low"],
            fill_value=0
        )
    )

    # Display department summary.
    print("\n==============================================")
    print("DEPARTMENT SUMMARY")
    print("==============================================")

    print(
        department_summary.to_string(index=False)
    )

    # Display top 10 requests.
    print("\n==============================================")
    print("TOP 10 MAINTENANCE REQUESTS")
    print("==============================================")

    display_columns = [
        "priority_rank",
        "request_id",
        "department",
        "maintenance_type",
        "operational_priority_score",
        "deadline_urgency_score",
        "final_priority_score",
        "final_priority_level",
        "hours_until_deadline"
    ]

    print(
        ranked_df[display_columns]
        .head(10)
        .to_string(index=False)
    )

    print("\n==============================================")
    print("PRIORITY ENGINE COMPLETED")
    print("==============================================")

    print("\nGenerated files:")
    print("-", OUTPUT_PATH)
    print("-", summary_path)

    print("\nScoring configuration:")
    print("Operational score weight:", OPERATIONAL_WEIGHT)
    print("Deadline urgency weight:", DEADLINE_WEIGHT)

    print("\nNOTE:")
    print(
        "This is a prototype ranking system using "
        "simulated planning data."
    )


if __name__ == "__main__":
    main()
