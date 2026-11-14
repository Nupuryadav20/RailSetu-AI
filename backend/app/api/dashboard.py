
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Query

from backend.app.services.artifact_service import (
    load_dataframe,
    load_csv,
    load_json,
)

from backend.app.services.filter_service import paginate_dataframe

from backend.app.services.model_service import predict_records
from backend.app.services.risk_service import score_records

from backend.app.schemas.prediction import (
    PredictionRequest,
    RiskScoringRequest,
)


router = APIRouter(prefix="/api", tags=["Dashboard"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = PROJECT_ROOT / "backend" / "artifacts"


# -------------------- DASHBOARD --------------------

@router.get("/dashboard/summary")
def dashboard_summary():

    kpis = load_json("planning/planning_kpis.json")

    scheduled = load_csv(
        "optimization/optimized_maintenance_schedule.csv"
    )

    unscheduled = load_csv(
        "optimization/unscheduled_maintenance_requests.csv"
    )

    requests = load_csv(
        "simulation/simulated_maintenance_requests.csv"
    )

    return {
        "project": "RailSetu AI",
        "total_maintenance_requests": len(requests),
        "scheduled_requests": len(scheduled),
        "unscheduled_requests": len(unscheduled),
        "planning_kpis": kpis,
        "data_source": "Simulated railway planning data",
    }


# -------------------- MAINTENANCE APIs --------------------

@router.get("/maintenance/requests")
def maintenance_requests(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str | None = Query(None, max_length=100),
    department: str | None = Query(None, max_length=100),
    priority: str | None = Query(None, max_length=50),
    corridor: str | None = Query(None, max_length=100),
):

    df = load_dataframe(
        "simulation/simulated_maintenance_requests.csv"
    )

    return paginate_dataframe(
        df, page, page_size,
        search, department, priority, corridor
    )


@router.get("/maintenance/priority")
def maintenance_priority(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str | None = Query(None, max_length=100),
    department: str | None = Query(None, max_length=100),
    priority: str | None = Query(None, max_length=50),
    corridor: str | None = Query(None, max_length=100),
):

    df = load_dataframe(
        "priority/ranked_maintenance_requests.csv"
    )

    return paginate_dataframe(
        df, page, page_size,
        search, department, priority, corridor
    )


# -------------------- SCHEDULE APIs --------------------

@router.get("/schedule/optimized")
def optimized_schedule():
    return load_csv(
        "optimization/optimized_maintenance_schedule.csv"
    )


@router.get("/schedule/unscheduled")
def unscheduled_requests():
    return load_csv(
        "optimization/unscheduled_maintenance_requests.csv"
    )


# -------------------- PLANNING APIs --------------------

@router.get("/planning/weekly")
def weekly_plan():
    return load_csv(
        "planning/weekly_maintenance_plan.csv"
    )


@router.get("/planning/monthly")
def monthly_plan():
    return load_csv(
        "planning/monthly_maintenance_plan.csv"
    )


@router.get("/planning/departments")
def department_summary():
    return {
        "weekly": load_csv(
            "planning/weekly_department_summary.csv"
        ),
        "monthly": load_csv(
            "planning/monthly_department_summary.csv"
        ),
    }


@router.get("/planning/corridors")
def corridor_summary():
    return {
        "weekly": load_csv(
            "planning/weekly_corridor_summary.csv"
        ),
        "monthly": load_csv(
            "planning/monthly_corridor_summary.csv"
        ),
    }


# -------------------- MACHINE LEARNING APIs --------------------

@router.get("/ml/model-comparison")
def model_comparison():
    return load_csv("metrics/model_comparison.csv")


@router.get("/ml/best-model")
def best_model():
    return load_json("metrics/best_model.json")


@router.get("/ml/feature-importance")
def feature_importance():
    return load_csv(
        "explainability/shap_feature_importance.csv"
    )


@router.get("/ml/prediction-explanations")
def prediction_explanations():
    return load_csv(
        "explainability/individual_prediction_explanation.csv"
    )


@router.post("/ml/predict")
def predict_maintenance(payload: PredictionRequest):

    records = [
        record.model_dump()
        for record in payload.records
    ]

    return predict_records(records)


@router.post("/risk/score")
def calculate_maintenance_risk(payload: RiskScoringRequest):

    records = [
        record.model_dump()
        for record in payload.records
    ]

    return score_records(records)


# -------------------- OPTIMIZATION APIs --------------------

@router.get("/optimization/validation")
def optimization_validation():
    return load_csv(
        "optimization/schedule_validation_report.csv"
    )


# -------------------- SIMULATION APIs --------------------

@router.get("/simulation/block-windows")
def block_windows():
    return load_csv(
        "simulation/simulated_block_windows.csv"
    )
