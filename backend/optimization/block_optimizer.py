
import os
import pandas as pd

from ortools.sat.python import cp_model


# ============================================================
# CONFIGURATION
# ============================================================

REQUESTS_PATH = (
    "backend/artifacts/priority/"
    "ranked_maintenance_requests.csv"
)

WINDOWS_PATH = (
    "backend/artifacts/simulation/"
    "simulated_block_windows.csv"
)

OUTPUT_DIR = "backend/artifacts/optimization"

SCHEDULE_PATH = os.path.join(
    OUTPUT_DIR,
    "optimized_maintenance_schedule.csv"
)

UNSCHEDULED_PATH = os.path.join(
    OUTPUT_DIR,
    "unscheduled_maintenance_requests.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "optimization_summary.csv"
)

# Simulated workforce capacity.
# These values are prototype assumptions, not actual railway rosters.
DEPARTMENT_WORKFORCE_CAPACITY = {
    "Engineering": 12,
    "Traction": 12,
    "Signal & Telecom": 12
}

SOLVER_TIME_LIMIT_SECONDS = 60


# ============================================================
# LOAD AND VALIDATE INPUTS
# ============================================================

def load_inputs():

    print("\nLoading maintenance requests and block windows...")

    requests = pd.read_csv(REQUESTS_PATH)
    windows = pd.read_csv(WINDOWS_PATH)

    required_request_columns = [
        "request_id",
        "corridor_id",
        "department",
        "maintenance_type",
        "required_duration_hours",
        "team_size_required",
        "earliest_start",
        "deadline",
        "final_priority_score",
        "final_priority_level"
    ]

    required_window_columns = [
        "window_id",
        "corridor_id",
        "window_start",
        "window_end",
        "window_status",
        "max_parallel_teams"
    ]

    missing_requests = [
        col for col in required_request_columns
        if col not in requests.columns
    ]

    missing_windows = [
        col for col in required_window_columns
        if col not in windows.columns
    ]

    if missing_requests:
        raise ValueError(
            f"Missing request columns: {missing_requests}"
        )

    if missing_windows:
        raise ValueError(
            f"Missing window columns: {missing_windows}"
        )

    requests["earliest_start"] = pd.to_datetime(
        requests["earliest_start"], errors="coerce"
    )

    requests["deadline"] = pd.to_datetime(
        requests["deadline"], errors="coerce"
    )

    windows["window_start"] = pd.to_datetime(
        windows["window_start"], errors="coerce"
    )

    windows["window_end"] = pd.to_datetime(
        windows["window_end"], errors="coerce"
    )

    requests["required_duration_hours"] = pd.to_numeric(
        requests["required_duration_hours"],
        errors="coerce"
    )

    requests["team_size_required"] = pd.to_numeric(
        requests["team_size_required"],
        errors="coerce"
    )

    requests["final_priority_score"] = pd.to_numeric(
        requests["final_priority_score"],
        errors="coerce"
    )

    requests = requests.dropna(
        subset=[
            "request_id",
            "corridor_id",
            "department",
            "earliest_start",
            "deadline",
            "required_duration_hours",
            "team_size_required",
            "final_priority_score"
        ]
    ).copy()

    windows = windows.dropna(
        subset=[
            "window_id",
            "corridor_id",
            "window_start",
            "window_end"
        ]
    ).copy()

    windows = windows[
        windows["window_status"].astype(str).str.lower()
        == "available"
    ].copy()

    if requests.empty:
        raise ValueError("No valid maintenance requests found.")

    if windows.empty:
        raise ValueError("No available block windows found.")

    if (requests["required_duration_hours"] <= 0).any():
        raise ValueError("Maintenance duration must be positive.")

    if (requests["team_size_required"] <= 0).any():
        raise ValueError("Required team size must be positive.")

    if (windows["window_end"] <= windows["window_start"]).any():
        raise ValueError("Invalid block window time range.")

    print("Valid maintenance requests:", len(requests))
    print("Available block windows:", len(windows))

    return requests, windows


# ============================================================
# BUILD OPTIMIZATION MODEL
# ============================================================

def optimize_schedule(requests, windows):

    print("\nBuilding OR-Tools optimization model...")

    model = cp_model.CpModel()

    # Use a common reference time.
    reference_time = min(
        requests["earliest_start"].min(),
        windows["window_start"].min()
    )

    def to_minutes(timestamp):
        return int(
            (timestamp - reference_time).total_seconds() // 60
        )

    # Store all candidate intervals.
    corridor_intervals = {}
    department_intervals = {}

    # Store candidate assignment variables.
    request_candidates = {}

    # Store candidate details for final output.
    candidate_details = {}

    # Objective function terms.
    objective_terms = []

    candidate_counter = 0

    # --------------------------------------------------------
    # Create feasible request-window combinations
    # --------------------------------------------------------

    for request_index, request in requests.iterrows():

        request_id = request["request_id"]
        corridor_id = request["corridor_id"]
        department = request["department"]

        duration_minutes = int(
            round(
                request["required_duration_hours"] * 60
            )
        )

        workforce_required = int(
            request["team_size_required"]
        )

        priority_score = float(
            request["final_priority_score"]
        )

        earliest = to_minutes(
            request["earliest_start"]
        )

        deadline = to_minutes(
            request["deadline"]
        )

        candidates_for_request = []

        department_capacity = (
            DEPARTMENT_WORKFORCE_CAPACITY.get(department)
        )

        if department_capacity is None:
            continue

        # A request requiring more workers than the entire
        # simulated department capacity cannot be scheduled.
        if workforce_required > department_capacity:
            continue

        matching_windows = windows[
            windows["corridor_id"] == corridor_id
        ]

        for _, window in matching_windows.iterrows():

            window_start = to_minutes(
                window["window_start"]
            )

            window_end = to_minutes(
                window["window_end"]
            )

            # Task must fit completely inside the window.
            earliest_feasible_start = max(
                earliest,
                window_start
            )

            latest_feasible_start = min(
                window_end - duration_minutes,
                deadline - duration_minutes
            )

            if (
                earliest_feasible_start
                > latest_feasible_start
            ):
                continue

            candidate_counter += 1

            candidate_id = (
                f"CAND_{candidate_counter}"
            )

            selected = model.NewBoolVar(
                f"selected_{candidate_id}"
            )

            start = model.NewIntVar(
                earliest_feasible_start,
                latest_feasible_start,
                f"start_{candidate_id}"
            )

            end = model.NewIntVar(
                earliest_feasible_start + duration_minutes,
                latest_feasible_start + duration_minutes,
                f"end_{candidate_id}"
            )

            interval = model.NewOptionalIntervalVar(
                start,
                duration_minutes,
                end,
                selected,
                f"interval_{candidate_id}"
            )

            candidates_for_request.append(selected)

            corridor_intervals.setdefault(
                corridor_id, []
            ).append(interval)

            department_intervals.setdefault(
                department, []
            ).append(
                (interval, workforce_required)
            )

            candidate_details[candidate_id] = {
                "request_index": request_index,
                "request_id": request_id,
                "window_id": window["window_id"],
                "corridor_id": corridor_id,
                "department": department,
                "duration_minutes": duration_minutes,
                "start_variable": start,
                "end_variable": end,
                "selection_variable": selected,
                "priority_score": priority_score
            }

            # Give higher-priority requests more weight.
            # A small time penalty favors earlier schedules.
            objective_terms.append(
                selected * int(priority_score * 10000)
                - start
            )

        request_candidates[request_index] = (
            candidates_for_request
        )

    print("Feasible request-window combinations:", candidate_counter)

    # --------------------------------------------------------
    # Each maintenance request can be selected at most once.
    # --------------------------------------------------------

    for request_index, candidates in request_candidates.items():

        if candidates:
            model.Add(
                sum(candidates) <= 1
            )

    # --------------------------------------------------------
    # No overlapping maintenance blocks on the same corridor.
    # --------------------------------------------------------

    print("Adding corridor non-overlap constraints...")

    for corridor_id, intervals in corridor_intervals.items():

        if intervals:
            model.AddNoOverlap(intervals)

    # --------------------------------------------------------
    # Department workforce capacity constraints.
    # --------------------------------------------------------

    print("Adding department workforce constraints...")

    for department, items in department_intervals.items():

        intervals = [item[0] for item in items]

        demands = [item[1] for item in items]

        capacity = DEPARTMENT_WORKFORCE_CAPACITY[
            department
        ]

        if intervals:
            model.AddCumulative(
                intervals,
                demands,
                capacity
            )

    # --------------------------------------------------------
    # Maximize priority-weighted scheduled work.
    # --------------------------------------------------------

    if not objective_terms:
        raise ValueError(
            "No feasible request-window combinations found."
        )

    model.Maximize(
        sum(objective_terms)
    )

    # ========================================================
    # SOLVE
    # ========================================================

    print("\nStarting CP-SAT solver...")

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = (
        SOLVER_TIME_LIMIT_SECONDS
    )

    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    print("\nSolver status:", solver.StatusName(status))

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE
    ):
        raise RuntimeError(
            "The optimizer could not find a feasible schedule."
        )

    # ========================================================
    # EXTRACT SCHEDULE
    # ========================================================

    scheduled_rows = []

    scheduled_request_ids = set()

    for candidate_id, details in candidate_details.items():

        selected = details["selection_variable"]

        if solver.Value(selected) != 1:
            continue

        start_minutes = solver.Value(
            details["start_variable"]
        )

        end_minutes = solver.Value(
            details["end_variable"]
        )

        start_time = (
            reference_time
            + pd.Timedelta(minutes=start_minutes)
        )

        end_time = (
            reference_time
            + pd.Timedelta(minutes=end_minutes)
        )

        request = requests.loc[
            details["request_index"]
        ]

        scheduled_request_ids.add(
            details["request_id"]
        )

        scheduled_rows.append({
            "schedule_id": f"SCH-{len(scheduled_rows) + 1:04d}",
            "request_id": details["request_id"],
            "window_id": details["window_id"],
            "corridor_id": details["corridor_id"],
            "asset_id": request.get("asset_id", ""),
            "location_km": request.get("location_km", ""),
            "department": details["department"],
            "maintenance_type": request["maintenance_type"],
            "priority_rank": request.get("priority_rank", ""),
            "priority_level": request["final_priority_level"],
            "priority_score": details["priority_score"],
            "required_duration_hours": (
                details["duration_minutes"] / 60
            ),
            "team_size_required": request["team_size_required"],
            "scheduled_start": start_time,
            "scheduled_end": end_time,
            "deadline": request["deadline"],
            "is_simulated": True
        })

    schedule = pd.DataFrame(scheduled_rows)

    # ========================================================
    # UNSCHEDULED REQUESTS
    # ========================================================

    unscheduled = requests[
        ~requests["request_id"].isin(
            scheduled_request_ids
        )
    ].copy()

    def explain_unscheduled(row):

        if row["request_id"] not in request_candidates:
            return "No feasible window: invalid or unsupported request."

        candidates = request_candidates[row.name]

        if not candidates:
            matching = windows[
                windows["corridor_id"] == row["corridor_id"]
            ]

            if matching.empty:
                return "No available window for this corridor."

            return (
                "No window fits the request duration, "
                "earliest start, and deadline."
            )

        return (
            "Not selected by optimizer due to corridor "
            "or department workforce capacity constraints "
            "or higher-priority competing requests."
        )

    unscheduled["unscheduled_reason"] = (
        unscheduled.apply(
            explain_unscheduled,
            axis=1
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_requests = len(requests)
    scheduled_count = len(schedule)
    unscheduled_count = len(unscheduled)

    if scheduled_count:
        total_work_hours = (
            schedule["required_duration_hours"].sum()
        )
    else:
        total_work_hours = 0

    summary = pd.DataFrame([{
        "solver_status": solver.StatusName(status),
        "total_requests": total_requests,
        "scheduled_requests": scheduled_count,
        "unscheduled_requests": unscheduled_count,
        "schedule_rate_percent": round(
            scheduled_count / total_requests * 100, 2
        ),
        "total_scheduled_work_hours": round(
            total_work_hours, 2
        ),
        "corridors_used": (
            schedule["corridor_id"].nunique()
            if scheduled_count else 0
        ),
        "simulated_workforce_capacities": str(
            DEPARTMENT_WORKFORCE_CAPACITY
        ),
        "is_simulated": True
    }])

    return schedule, unscheduled, summary


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(schedule, unscheduled, summary):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    schedule.to_csv(
        SCHEDULE_PATH,
        index=False
    )

    unscheduled.to_csv(
        UNSCHEDULED_PATH,
        index=False
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False
    )

    print("\nSaved output files:")
    print("-", SCHEDULE_PATH)
    print("-", UNSCHEDULED_PATH)
    print("-", SUMMARY_PATH)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==============================================")
    print("RAILSETU AI - BLOCK SCHEDULING OPTIMIZER")
    print("==============================================")

    requests, windows = load_inputs()

    schedule, unscheduled, summary = optimize_schedule(
        requests,
        windows
    )

    save_outputs(
        schedule,
        unscheduled,
        summary
    )

    print("\n==============================================")
    print("OPTIMIZATION RESULTS")
    print("==============================================")

    print(summary.to_string(index=False))

    print("\nDepartment-wise scheduled requests:")

    if not schedule.empty:
        print(
            schedule.groupby("department")
            .size()
            .to_string()
        )

        print("\nSample optimized schedule:")

        print(
            schedule.sort_values("scheduled_start")
            .head(15)
            .to_string(index=False)
        )

    print("\nUnscheduled requests:", len(unscheduled))

    print("\n==============================================")
    print("BLOCK SCHEDULING OPTIMIZATION COMPLETED")
    print("==============================================")

    print(
        "\nNOTE: This is a prototype optimization using "
        "simulated corridor windows and workforce capacities."
    )


if __name__ == "__main__":
    main()
