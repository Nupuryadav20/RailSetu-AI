
import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SCHEDULE_PATH = (
    "backend/artifacts/optimization/"
    "optimized_maintenance_schedule.csv"
)

WINDOWS_PATH = (
    "backend/artifacts/simulation/"
    "simulated_block_windows.csv"
)

OUTPUT_DIR = "backend/artifacts/optimization"

VALIDATION_REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "schedule_validation_report.csv"
)

DEPARTMENT_WORKFORCE_CAPACITY = {
    "Engineering": 12,
    "Traction": 12,
    "Signal & Telecom": 12
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(SCHEDULE_PATH):
        raise FileNotFoundError(
            f"Schedule not found: {SCHEDULE_PATH}. "
            "Run the block optimizer first."
        )

    if not os.path.exists(WINDOWS_PATH):
        raise FileNotFoundError(
            f"Block windows not found: {WINDOWS_PATH}"
        )

    schedule = pd.read_csv(SCHEDULE_PATH)
    windows = pd.read_csv(WINDOWS_PATH)

    if schedule.empty:
        raise ValueError(
            "The schedule is empty. Nothing to validate."
        )

    datetime_columns = [
        "scheduled_start",
        "scheduled_end",
        "deadline"
    ]

    for col in datetime_columns:
        schedule[col] = pd.to_datetime(
            schedule[col], errors="coerce"
        )

    windows["window_start"] = pd.to_datetime(
        windows["window_start"], errors="coerce"
    )

    windows["window_end"] = pd.to_datetime(
        windows["window_end"], errors="coerce"
    )

    return schedule, windows


# ============================================================
# VALIDATION
# ============================================================

def validate_schedule(schedule, windows):

    errors = []

    checks = {
        "unique_request_assignment": True,
        "valid_time_ranges": True,
        "duration_correct": True,
        "deadline_respected": True,
        "earliest_start_respected": True,
        "inside_assigned_window": True,
        "corridor_overlap_free": True,
        "department_capacity_respected": True
    }

    # --------------------------------------------------------
    # 1. Unique request assignment
    # --------------------------------------------------------

    duplicate_requests = schedule[
        schedule["request_id"].duplicated(keep=False)
    ]

    if not duplicate_requests.empty:
        checks["unique_request_assignment"] = False

        errors.append(
            f"Duplicate request assignments: "
            f"{duplicate_requests['request_id'].tolist()}"
        )

    # --------------------------------------------------------
    # 2. Valid time ranges
    # --------------------------------------------------------

    invalid_times = schedule[
        schedule["scheduled_start"].isna()
        | schedule["scheduled_end"].isna()
        | schedule["deadline"].isna()
        | (
            schedule["scheduled_end"]
            <= schedule["scheduled_start"]
        )
    ]

    if not invalid_times.empty:
        checks["valid_time_ranges"] = False

        errors.append(
            "Invalid or missing time values for requests: "
            + str(invalid_times["request_id"].tolist())
        )

    # --------------------------------------------------------
    # 3. Maintenance duration
    # --------------------------------------------------------

    calculated_duration = (
        schedule["scheduled_end"]
        - schedule["scheduled_start"]
    ).dt.total_seconds() / 3600

    incorrect_duration = schedule[
        (
            calculated_duration
            - schedule["required_duration_hours"]
        ).abs() > 0.01
    ]

    if not incorrect_duration.empty:
        checks["duration_correct"] = False

        errors.append(
            "Duration mismatch for requests: "
            + str(incorrect_duration["request_id"].tolist())
        )

    # --------------------------------------------------------
    # 4. Deadline
    # --------------------------------------------------------

    late_jobs = schedule[
        schedule["scheduled_end"] > schedule["deadline"]
    ]

    if not late_jobs.empty:
        checks["deadline_respected"] = False

        errors.append(
            "Deadline violations: "
            + str(late_jobs["request_id"].tolist())
        )

    # --------------------------------------------------------
    # 5. Earliest start
    # --------------------------------------------------------

    # The optimizer's input requests contain earliest_start.
    # Load the source file to verify it independently.

    requests_path = (
        "backend/artifacts/priority/"
        "ranked_maintenance_requests.csv"
    )

    requests = pd.read_csv(requests_path)

    requests["earliest_start"] = pd.to_datetime(
        requests["earliest_start"], errors="coerce"
    )

    earliest_lookup = requests.set_index(
        "request_id"
    )["earliest_start"].to_dict()

    earliest_violations = []

    for _, row in schedule.iterrows():

        earliest = earliest_lookup.get(
            row["request_id"]
        )

        if pd.isna(earliest):
            earliest_violations.append(
                row["request_id"]
            )
            continue

        if row["scheduled_start"] < earliest:
            earliest_violations.append(
                row["request_id"]
            )

    if earliest_violations:
        checks["earliest_start_respected"] = False

        errors.append(
            "Earliest-start violations: "
            + str(earliest_violations)
        )

    # --------------------------------------------------------
    # 6. Assigned block window
    # --------------------------------------------------------

    window_lookup = windows.set_index(
        "window_id"
    ).to_dict("index")

    window_violations = []

    for _, row in schedule.iterrows():

        window = window_lookup.get(
            row["window_id"]
        )

        if window is None:
            window_violations.append(
                row["request_id"]
            )
            continue

        if (
            row["corridor_id"] != window["corridor_id"]
            or row["scheduled_start"] < window["window_start"]
            or row["scheduled_end"] > window["window_end"]
            or str(window["window_status"]).lower()
            != "available"
        ):
            window_violations.append(
                row["request_id"]
            )

    if window_violations:
        checks["inside_assigned_window"] = False

        errors.append(
            "Block-window violations: "
            + str(window_violations)
        )

    # --------------------------------------------------------
    # 7. Corridor overlap
    # --------------------------------------------------------

    corridor_conflicts = []

    for corridor, group in schedule.groupby("corridor_id"):

        group = group.sort_values("scheduled_start")

        previous_end = None
        previous_request = None

        for _, row in group.iterrows():

            if (
                previous_end is not None
                and row["scheduled_start"] < previous_end
            ):
                corridor_conflicts.append(
                    (
                        corridor,
                        previous_request,
                        row["request_id"]
                    )
                )

            if (
                previous_end is None
                or row["scheduled_end"] > previous_end
            ):
                previous_end = row["scheduled_end"]
                previous_request = row["request_id"]

    if corridor_conflicts:
        checks["corridor_overlap_free"] = False

        errors.append(
            "Corridor overlap conflicts: "
            + str(corridor_conflicts)
        )

    # --------------------------------------------------------
    # 8. Department workforce capacity
    # --------------------------------------------------------

    workforce_conflicts = []

    for department, group in schedule.groupby("department"):

        capacity = DEPARTMENT_WORKFORCE_CAPACITY.get(
            department
        )

        if capacity is None:
            workforce_conflicts.append(
                f"Unknown department: {department}"
            )
            continue

        events = []

        for _, row in group.iterrows():

            start = row["scheduled_start"]
            end = row["scheduled_end"]
            team_size = int(row["team_size_required"])

            # End events sort before start events at the
            # same timestamp because intervals are half-open.
            events.append((start, 1, team_size, row["request_id"]))
            events.append((end, 0, -team_size, row["request_id"]))

        events.sort(key=lambda item: (item[0], item[1]))

        active_teams = 0

        for timestamp, event_type, change, request_id in events:

            active_teams += change

            if active_teams > capacity:
                workforce_conflicts.append(
                    (
                        department,
                        timestamp,
                        active_teams,
                        capacity
                    )
                )
                break

    if workforce_conflicts:
        checks["department_capacity_respected"] = False

        errors.append(
            "Department workforce capacity conflicts: "
            + str(workforce_conflicts)
        )

    return checks, errors


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==============================================")
    print("RAILSETU AI - SCHEDULE VALIDATION")
    print("==============================================")

    schedule, windows = load_data()

    print("\nScheduled requests:", len(schedule))
    print("Available windows:", len(windows))

    checks, errors = validate_schedule(
        schedule,
        windows
    )

    print("\nVALIDATION RESULTS")
    print("----------------------------------------------")

    report_rows = []

    for check_name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(f"{check_name}: {status}")

        report_rows.append({
            "validation_check": check_name,
            "status": status
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pd.DataFrame(report_rows).to_csv(
        VALIDATION_REPORT_PATH,
        index=False
    )

    print("\n----------------------------------------------")

    if all(checks.values()):

        print("ALL VALIDATION CHECKS PASSED")

    else:

        print("VALIDATION FAILED")

        for error in errors:
            print("\nERROR:", error)

    print("\nValidation report saved:")
    print(VALIDATION_REPORT_PATH)

    print("\nNOTE:")
    print(
        "This validates the prototype's modeled constraints. "
        "It does not certify railway operational safety."
    )

    print("\n==============================================")


if __name__ == "__main__":
    main()
