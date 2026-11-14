
import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "backend/artifacts/simulation"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "simulated_block_windows.csv"
)

CORRIDORS = [
    "CORRIDOR-A",
    "CORRIDOR-B",
    "CORRIDOR-C",
    "CORRIDOR-D",
    "CORRIDOR-E"
]

PLANNING_HORIZON_DAYS = 14

# Simulated maintenance windows.
# Each window represents a period when the corridor
# is assumed to be available for maintenance.

WINDOWS = [
    ("00:00", "06:00"),
    ("12:00", "18:00")
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# GENERATE BLOCK WINDOWS
# ============================================================

def generate_windows():

    today = pd.Timestamp.now().normalize()

    records = []

    window_counter = 1

    print("\nGenerating simulated corridor availability...")

    for day in range(PLANNING_HORIZON_DAYS):

        current_date = today + pd.Timedelta(days=day)

        for corridor in CORRIDORS:

            for start_time, end_time in WINDOWS:

                window_start = pd.Timestamp(
                    f"{current_date.date()} {start_time}"
                )

                window_end = pd.Timestamp(
                    f"{current_date.date()} {end_time}"
                )

                duration = (
                    window_end - window_start
                ).total_seconds() / 3600

                records.append({
                    "window_id": f"WIN-{window_counter:04d}",
                    "corridor_id": corridor,
                    "window_start": window_start,
                    "window_end": window_end,
                    "available_duration_hours": duration,
                    "max_parallel_teams": 1,
                    "window_status": "Available",
                    "availability_source": "Simulated Timetable",
                    "is_simulated": True
                })

                window_counter += 1

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==============================================")
    print("RAILSETU AI - BLOCK WINDOW GENERATOR")
    print("==============================================")

    df = generate_windows()

    # Save timestamps in readable format.
    df["window_start"] = pd.to_datetime(
        df["window_start"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    df["window_end"] = pd.to_datetime(
        df["window_end"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nTotal corridors:", len(CORRIDORS))
    print("Planning horizon:", PLANNING_HORIZON_DAYS, "days")
    print("Windows per corridor per day:", len(WINDOWS))
    print("Total available windows:", len(df))

    print("\nCorridor-wise window count:")
    print(
        df.groupby("corridor_id")
        .size()
        .to_string()
    )

    print("\nSample block windows:")
    print(df.head(10).to_string(index=False))

    print("\n==============================================")
    print("BLOCK WINDOW GENERATION COMPLETED")
    print("==============================================")

    print("\nSaved file:")
    print(OUTPUT_PATH)

    print("\nIMPORTANT:")
    print(
        "These availability windows are simulated. "
        "They do not represent actual railway timetable data."
    )


if __name__ == "__main__":
    main()
