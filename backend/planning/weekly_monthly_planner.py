import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("backend/artifacts")

SCHEDULE_PATH = (
    BASE_DIR / "optimization" /
    "optimized_maintenance_schedule.csv"
)

UNSCHEDULED_PATH = (
    BASE_DIR / "optimization" /
    "unscheduled_maintenance_requests.csv"
)

OUTPUT_DIR = BASE_DIR / "planning"

REQUIRED_COLUMNS = [
    "schedule_id",
    "request_id",
    "corridor_id",
    "department",
    "maintenance_type",
    "scheduled_start",
    "scheduled_end",
    "priority_level",
    "priority_score",
    "required_duration_hours",
    "team_size_required",
]


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_schedule():

    print("\nLoading optimized maintenance schedule...")

    if not SCHEDULE_PATH.exists():
        raise FileNotFoundError(
            f"Schedule not found: {SCHEDULE_PATH}\n"
            "Run the block optimizer first:\n"
            "python -m backend.optimization.block_optimizer"
        )

    schedule = pd.read_csv(SCHEDULE_PATH)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in schedule.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required schedule columns: {missing}"
        )

    if schedule.empty:
        raise ValueError("The optimized schedule is empty.")

    schedule["scheduled_start"] = pd.to_datetime(
        schedule["scheduled_start"], errors="coerce"
    )

    schedule["scheduled_end"] = pd.to_datetime(
        schedule["scheduled_end"], errors="coerce"
    )

    schedule["required_duration_hours"] = pd.to_numeric(
        schedule["required_duration_hours"],
        errors="coerce"
    )

    schedule["team_size_required"] = pd.to_numeric(
        schedule["team_size_required"],
        errors="coerce"
    )

    schedule["priority_score"] = pd.to_numeric(
        schedule["priority_score"],
        errors="coerce"
    )

    schedule = schedule.dropna(
        subset=[
            "request_id",
            "scheduled_start",
            "scheduled_end",
            "department",
            "corridor_id",
        ]
    ).copy()

    if schedule.empty:
        raise ValueError(
            "No valid scheduled maintenance records remain."
        )

    if (schedule["scheduled_end"] <=
            schedule["scheduled_start"]).any():
        raise ValueError(
            "Found a maintenance job with an invalid time range."
        )

    if schedule["request_id"].duplicated().any():
        raise ValueError(
            "Duplicate request assignments found in the schedule."
        )

    # Recalculate duration from actual scheduled timestamps.
    schedule["actual_scheduled_hours"] = (
        schedule["scheduled_end"] -
        schedule["scheduled_start"]
    ).dt.total_seconds() / 3600

    # Weekly planning periods start on Monday.
    schedule["week_start"] = (
        schedule["scheduled_start"].dt.normalize()
        - pd.to_timedelta(
            schedule["scheduled_start"].dt.weekday,
            unit="D"
        )
    )

    schedule["week_end"] = (
        schedule["week_start"] + pd.Timedelta(days=6)
    )

    # Monthly planning periods.
    schedule["month_start"] = (
        schedule["scheduled_start"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    schedule["month_end"] = (
        schedule["month_start"]
        + pd.offsets.MonthEnd(0)
    )

    # Identify tasks crossing reporting boundaries.
    schedule["crosses_week_boundary"] = (
        schedule["scheduled_end"].dt.normalize()
        > schedule["week_end"]
    )

    schedule["crosses_month_boundary"] = (
        schedule["scheduled_end"].dt.to_period("M")
        != schedule["scheduled_start"].dt.to_period("M")
    )

    schedule["is_simulated"] = True

    print("Valid scheduled jobs:", len(schedule))
    print(
        "Planning date range:",
        schedule["scheduled_start"].min(),
        "to",
        schedule["scheduled_end"].max()
    )

    return schedule


# ============================================================
# CREATE WEEKLY PLAN
# ============================================================

def create_weekly_plan(schedule):

    print("\nGenerating weekly maintenance plan...")

    weekly = schedule.copy()

    weekly["planning_period"] = (
        weekly["week_start"].dt.strftime("%Y-%m-%d")
        + " to "
        + weekly["week_end"].dt.strftime("%Y-%m-%d")
    )

    weekly = weekly.sort_values(
        ["week_start", "scheduled_start", "priority_score"],
        ascending=[True, True, False]
    )

    return weekly


# ============================================================
# CREATE MONTHLY PLAN
# ============================================================

def create_monthly_plan(schedule):

    print("Generating monthly maintenance plan...")

    monthly = schedule.copy()

    monthly["planning_period"] = (
        monthly["month_start"].dt.strftime("%B %Y")
    )

    monthly = monthly.sort_values(
        ["month_start", "scheduled_start", "priority_score"],
        ascending=[True, True, False]
    )

    return monthly


# ============================================================
# GENERATE SUMMARY REPORTS
# ============================================================

def create_summary(data, period_column, group_columns):

    summary = (
        data.groupby(
            [period_column] + group_columns,
            dropna=False
        )
        .agg(
            total_jobs=("request_id", "nunique"),
            total_work_hours=("actual_scheduled_hours", "sum"),
            total_workers_required=("team_size_required", "sum"),
            average_priority_score=("priority_score", "mean"),
            high_priority_jobs=(
                "priority_level",
                lambda values: values.astype(str)
                .str.lower().eq("high").sum()
            ),
            critical_priority_jobs=(
                "priority_level",
                lambda values: values.astype(str)
                .str.lower().eq("critical").sum()
            ),
            first_job=("scheduled_start", "min"),
            last_job=("scheduled_end", "max"),
        )
        .reset_index()
    )

    summary["total_work_hours"] = (
        summary["total_work_hours"].round(2)
    )

    summary["average_priority_score"] = (
        summary["average_priority_score"].round(2)
    )

    return summary


# ============================================================
# GENERATE OVERALL PLANNING KPIs
# ============================================================

def create_kpis(schedule, unscheduled_count):

    total_jobs = len(schedule)

    total_hours = float(
        schedule["actual_scheduled_hours"].sum()
    )

    kpis = {
        "project": "RailSetu AI",
        "planning_type": "Weekly and Monthly Maintenance Planning",
        "is_simulated": True,
        "scheduled_jobs": int(total_jobs),
        "unscheduled_jobs": int(unscheduled_count),
        "total_scheduled_work_hours": round(total_hours, 2),
        "departments_involved": int(
            schedule["department"].nunique()
        ),
        "corridors_involved": int(
            schedule["corridor_id"].nunique()
        ),
        "weekly_periods": int(
            schedule["week_start"].nunique()
        ),
        "monthly_periods": int(
            schedule["month_start"].nunique()
        ),
        "high_priority_jobs": int(
            schedule["priority_level"]
            .astype(str).str.lower().eq("high").sum()
        ),
        "critical_priority_jobs": int(
            schedule["priority_level"]
            .astype(str).str.lower().eq("critical").sum()
        ),
        "jobs_crossing_week_boundary": int(
            schedule["crosses_week_boundary"].sum()
        ),
        "jobs_crossing_month_boundary": int(
            schedule["crosses_month_boundary"].sum()
        ),
        "planning_start": str(
            schedule["scheduled_start"].min()
        ),
        "planning_end": str(
            schedule["scheduled_end"].max()
        ),
        "note": (
            "Prototype planning output based on simulated "
            "maintenance requests, corridor windows, and "
            "workforce capacities. Not an official railway plan."
        ),
    }

    return kpis


# ============================================================
# SAVE ALL REPORTS
# ============================================================

def save_reports(
    weekly,
    monthly,
    weekly_department,
    monthly_department,
    weekly_corridor,
    monthly_corridor,
    kpis,
    unscheduled
):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    weekly.to_csv(
        OUTPUT_DIR / "weekly_maintenance_plan.csv",
        index=False
    )

    monthly.to_csv(
        OUTPUT_DIR / "monthly_maintenance_plan.csv",
        index=False
    )

    weekly_department.to_csv(
        OUTPUT_DIR / "weekly_department_summary.csv",
        index=False
    )

    monthly_department.to_csv(
        OUTPUT_DIR / "monthly_department_summary.csv",
        index=False
    )

    weekly_corridor.to_csv(
        OUTPUT_DIR / "weekly_corridor_summary.csv",
        index=False
    )

    monthly_corridor.to_csv(
        OUTPUT_DIR / "monthly_corridor_summary.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / "planning_kpis.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(kpis, file, indent=4)

    if unscheduled is not None:
        unscheduled.to_csv(
            OUTPUT_DIR / "unscheduled_requests_for_planning.csv",
            index=False
        )

    print("\nAll planning reports saved in:")
    print(OUTPUT_DIR)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==============================================")
    print("RAILSETU AI - WEEKLY AND MONTHLY PLANNING")
    print("==============================================")

    schedule = load_schedule()

    weekly = create_weekly_plan(schedule)
    monthly = create_monthly_plan(schedule)

    weekly_department = create_summary(
        weekly,
        "planning_period",
        ["department"]
    )

    monthly_department = create_summary(
        monthly,
        "planning_period",
        ["department"]
    )

    weekly_corridor = create_summary(
        weekly,
        "planning_period",
        ["corridor_id"]
    )

    monthly_corridor = create_summary(
        monthly,
        "planning_period",
        ["corridor_id"]
    )

    unscheduled = None
    unscheduled_count = 0

    if UNSCHEDULED_PATH.exists():
        unscheduled = pd.read_csv(UNSCHEDULED_PATH)
        unscheduled_count = len(unscheduled)

    kpis = create_kpis(
        schedule,
        unscheduled_count
    )

    save_reports(
        weekly,
        monthly,
        weekly_department,
        monthly_department,
        weekly_corridor,
        monthly_corridor,
        kpis,
        unscheduled
    )

    print("\n==============================================")
    print("PLANNING RESULTS")
    print("==============================================")

    print("Scheduled jobs:", kpis["scheduled_jobs"])
    print("Unscheduled jobs:", kpis["unscheduled_jobs"])
    print("Total work hours:", kpis["total_scheduled_work_hours"])
    print("Weekly periods:", kpis["weekly_periods"])
    print("Monthly periods:", kpis["monthly_periods"])

    print("\nWeekly department summary:")
    print(weekly_department.to_string(index=False))

    print("\nMonthly department summary:")
    print(monthly_department.to_string(index=False))

    print("\n==============================================")
    print("WEEKLY AND MONTHLY PLANNING COMPLETED")
    print("==============================================")

    print(
        "\nNOTE: These reports are prototype planning outputs "
        "based on simulated railway planning data."
    )


if __name__ == "__main__":
    main()