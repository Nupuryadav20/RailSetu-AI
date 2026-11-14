
from pathlib import Path
import json

import pandas as pd
from fastapi import HTTPException


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = PROJECT_ROOT / "backend" / "artifacts"
REPORTS = PROJECT_ROOT / "reports"


def get_artifact_path(filename):
    file_path = ARTIFACTS / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Artifact not found: {filename}"
        )

    return file_path


def get_report_path(filename):
    file_path = REPORTS / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report file not found: {filename}"
        )
    return file_path


def load_report_csv(filename):
    """Load report CSV as JSON-safe records."""
    file_path = get_report_path(filename)
    try:
        df = pd.read_csv(file_path)
        return json.loads(df.to_json(orient="records", date_format="iso"))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read report CSV: {filename}"
        ) from exc


def load_report_json(filename):
    """Load report JSON artifact."""
    file_path = get_report_path(filename)
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read report JSON: {filename}"
        ) from exc


def load_dataframe(filename):
    """Load a CSV artifact as a Pandas DataFrame."""
    file_path = get_artifact_path(filename)

    try:
        return pd.read_csv(file_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read artifact: {filename}"
        ) from exc


def load_csv(filename):
    """Load CSV data as JSON-safe records."""
    df = load_dataframe(filename)

    return json.loads(
        df.to_json(
            orient="records",
            date_format="iso"
        )
    )


def load_json(filename):
    """Load a JSON artifact."""
    file_path = get_artifact_path(filename)

    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read JSON artifact: {filename}"
        ) from exc
