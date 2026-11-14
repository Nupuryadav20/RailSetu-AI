
from pathlib import Path
import json

import pandas as pd
from fastapi import APIRouter, Query, HTTPException, Response
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.database.connection import SessionLocal
from backend.app.database.models import FeatureStat, ModelRun, ModelMetric

from backend.app.services.artifact_service import (
    load_dataframe,
    load_csv,
    load_json,
    load_report_csv,
    load_report_json,
    get_report_path,
)

from backend.app.services.filter_service import paginate_dataframe

from backend.app.services.model_service import (
    predict_records,
    get_prediction_history,
    export_predictions_csv,
)
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


# -------------------- V4 METRICS & REPORTS (P2/P3) --------------------

@router.get("/ml/metrics/full")
def ml_metrics_full():
    return load_report_csv("v4/master_metrics.csv")


@router.get("/ml/metrics/val")
def ml_metrics_val():
    return load_report_csv("v4/val_model_comparison.csv")


@router.get("/ml/metrics/test")
def ml_metrics_test():
    return load_report_csv("v4/test_model_comparison.csv")


@router.get("/ml/threshold-analysis")
def ml_threshold_analysis():
    return load_report_csv("v4/threshold_analysis.csv")


@router.get("/ml/cv-summary")
def ml_cv_summary():
    return load_report_json("v4/cv_summary.json")


@router.get("/ml/seed-stability")
def ml_seed_stability():
    return load_report_csv("v4/seed_stability.csv")


@router.get("/ml/permutation-importance")
def ml_permutation_importance():
    return load_report_csv("v4/permutation_importance.csv")


@router.get("/ml/shap/v4")
def ml_shap_v4():
    return load_report_csv("v4/shap_feature_importance_v4.csv")


@router.get("/ml/shap/examples")
def ml_shap_examples():
    return load_report_json("v4/shap_worked_examples.json")


@router.get("/ml/metadata")
def ml_metadata():
    return load_report_json("v4/v4_metadata.json")


@router.get("/ml/plots/{name}")
def ml_plot(name: str):
    # Sanitize name to avoid traversal
    safe_name = Path(name).name
    if not safe_name.endswith(".png"):
        safe_name = f"{safe_name}.png"
    plot_file = get_report_path(f"v4/{safe_name}")
    return FileResponse(plot_file, media_type="image/png")


@router.get("/ml/plots/v5/{name}")
def ml_plot_v5(name: str):
    safe_name = Path(name).name
    if not safe_name.endswith(".png"):
        safe_name = f"{safe_name}.png"
    plot_file = get_report_path(f"v5/{safe_name}")
    return FileResponse(plot_file, media_type="image/png")


# -------------------- V5 METRICS & EXPERIMENTS (TASK 4) --------------------

@router.get("/ml/v5/comparison")
def ml_v5_comparison(split: str = Query("test", pattern="^(test|val)$")):
    if split == "val":
        return load_report_csv("v5/val_model_comparison_v5.csv")
    return load_report_csv("v5/test_model_comparison_v5.csv")


@router.get("/ml/v5/metrics/{model}")
def ml_v5_model_metrics(model: str):
    records = load_report_csv("v5/master_metrics_v5.csv")
    filtered = [r for r in records if r.get("model", "").lower() == model.lower()]
    if not filtered:
        raise HTTPException(status_code=404, detail=f"Metrics for model '{model}' not found in v5 master metrics.")
    return filtered


@router.get("/ml/v5/thresholds")
def ml_v5_thresholds():
    return load_report_csv("v5/threshold_analysis_v5.csv")


@router.get("/ml/v5/cv")
def ml_v5_cv_summary():
    return load_report_json("v5/cv_summary_v5.json")


@router.get("/ml/v5/feature-selection")
def ml_v5_feature_selection():
    return {
        "ranking": load_report_csv("v5/feature_ranking_v5.csv"),
        "ablation": load_report_csv("v5/feature_selection_ablation.csv"),
    }


@router.get("/ml/v5/feature-stats")
def ml_v5_feature_stats():
    db: Session = SessionLocal()
    try:
        stats = db.query(FeatureStat).all()
        out = []
        for s in stats:
            out.append({
                "feature": s.feature,
                "data_type": s.data_type,
                "min": s.min_val,
                "max": s.max_val,
                "mean": s.mean_val,
                "std": s.std_val,
                "p25": s.p25,
                "p50": s.p50,
                "p75": s.p75,
                "class_0_mean": s.class_0_mean,
                "class_1_mean": s.class_1_mean,
                "categories": json.loads(s.categories_json) if s.categories_json else [],
            })
        return out
    finally:
        db.close()


@router.get("/ml/v5/experiments")
def ml_v5_experiments():
    db: Session = SessionLocal()
    try:
        runs = db.query(ModelRun).order_by(ModelRun.id.desc()).all()
        out = []
        for r in runs:
            out.append({
                "id": r.id,
                "mlflow_run_id": r.mlflow_run_id,
                "model_name": r.model_name,
                "version": r.version,
                "params": json.loads(r.params_json) if r.params_json else {},
                "created_at": r.created_at,
                "is_served": r.is_served,
            })
        return out
    finally:
        db.close()


@router.get("/ml/v5/metadata")
def ml_v5_metadata():
    return load_report_json("v5/v5_metadata.json")


@router.get("/ml/predictions")
def ml_predictions(
    limit: int = Query(50, ge=1, le=500),
    model_version: str | None = Query(None),
    predicted_class: int | None = Query(None),
    min_prob: float | None = Query(None),
    max_prob: float | None = Query(None),
):
    return get_prediction_history(
        limit=limit,
        model_version=model_version,
        predicted_class=predicted_class,
        min_prob=min_prob,
        max_prob=max_prob,
    )


@router.get("/ml/predictions/download")
def ml_predictions_download(
    limit: int = Query(500, ge=1, le=5000),
    model_version: str | None = Query(None),
):
    csv_data = export_predictions_csv(limit=limit, model_version=model_version)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=railsetu_predictions.csv"},
    )


@router.get("/predict/history")
def predict_history():
    return get_prediction_history()


@router.get("/dataset/stats")
def dataset_stats():
    return {
        "dataset_name": "Indian Railway Predictive Maintenance (100K)",
        "source": "Synthetic / simulated maintenance records (Kaggle)",
        "is_live_railway_data": False,
        "total_records": 100000,
        "total_features": 35,
        "ml_features_count": 30,
        "numerical_features_count": 25,
        "categorical_features_count": 5,
        "target_column": "maintenance_required",
        "target_distribution": {
            "class_0_negative": 79804,
            "class_1_positive": 20196,
            "positive_rate_percent": 20.2,
            "imbalance_ratio": "3.95:1"
        },
        "leakage_excluded_columns": [
            "risk_score",
            "failure_type",
            "failure_severity",
            "train_id"
        ],
        "splits": {
            "train": 60000,
            "val": 20000,
            "test": 20000
        },
        "note": "Decision-support prototype only. Target has weak signal (ROC-AUC ~0.55)."
    }


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
