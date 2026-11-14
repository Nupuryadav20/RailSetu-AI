import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.main import app
from backend.app.database.models import (
    Base,
    MaintenanceRequest,
    BlockWindow,
    ScheduleAssignment,
    UnscheduledRequest,
    Prediction,
    ModelRun,
    FeatureStat,
)
from backend.app.services.model_service import load_model_bundle

client = TestClient(app)


def test_v5_comparison_test_split():
    resp = client.get("/api/ml/v5/comparison")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 8
    models = [r.get("model") for r in data]
    assert "XGBoost" in models
    assert "LightGBM" in models
    assert "Dummy" in models


def test_v5_comparison_val_split():
    resp = client.get("/api/ml/v5/comparison?split=val")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 8


def test_v5_model_metrics():
    resp = client.get("/api/ml/v5/metrics/XGBoost")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    splits = {r["split"] for r in data}
    assert {"train", "val", "test"}.issubset(splits)


def test_v5_thresholds():
    resp = client.get("/api/ml/v5/thresholds")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    thresholds = [r["threshold"] for r in data]
    assert 0.45 in thresholds
    assert 0.1 in thresholds


def test_v5_cv_summary():
    resp = client.get("/api/ml/v5/cv")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "XGBoost"
    assert "cv_roc_auc_mean" in data
    assert "cv_pr_auc_mean" in data
    assert len(data.get("scores_pr", [])) == 10


def test_v5_feature_selection():
    resp = client.get("/api/ml/v5/feature-selection")
    assert resp.status_code == 200
    data = resp.json()
    assert "ranking" in data
    assert "ablation" in data
    assert len(data["ranking"]) == 30
    assert len(data["ablation"]) >= 7


def test_v5_feature_stats():
    resp = client.get("/api/ml/v5/feature-stats")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 30
    first = data[0]
    assert "feature" in first
    assert "min" in first
    assert "max" in first
    assert "mean" in first
    assert "std" in first
    assert "class_0_mean" in first
    assert "class_1_mean" in first


def test_v5_experiments():
    resp = client.get("/api/ml/v5/experiments")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 2
    versions = [r["version"] for r in data]
    assert "v5" in versions


def test_v5_metadata():
    resp = client.get("/api/ml/v5/metadata")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == "v5"
    assert data["selected_features_k"] == 30
    assert "switch_api_to_v5" in data


def test_v5_predict_rich_diagnostics_and_persistence():
    x_test = pd.read_csv("data/processed/model_ready/X_test_v4.csv")
    sample_row = x_test.iloc[0].to_dict()

    resp = client.post("/api/ml/predict", json={"records": [sample_row]})
    assert resp.status_code == 200
    data = resp.json()

    assert data["model_version"] == "v5"
    assert data["is_calibrated"] is True
    assert "results" in data
    res = data["results"][0]

    assert "maintenance_probability_percent" in res
    assert "calibrated_probability_percent" in res
    assert "diagnostics" in res
    assert len(res["diagnostics"]) == 30

    d0 = res["diagnostics"][0]
    assert "feature" in d0
    assert "z_score" in d0
    assert "percentile_rank" in d0
    assert "class_0_mean" in d0
    assert "class_1_mean" in d0
    assert "top_shap_contributors" in res

    # Verify query predictions returns the stored record
    history_resp = client.get("/api/ml/predictions?limit=5")
    assert history_resp.status_code == 200
    hist = history_resp.json()
    assert len(hist) > 0
    assert hist[0]["model_version"] == "v5"


def test_v5_predictions_csv_download():
    resp = client.get("/api/ml/predictions/download?limit=10")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    content = resp.text
    assert "timestamp" in content
    assert "model_version" in content


def test_feature_parity_training_vs_inference():
    """Verify that training features, preprocessor features, and metadata match strictly."""
    bundle = load_model_bundle()
    pipe, meta, threshold, version = bundle
    assert version == "v5"

    x_train = pd.read_csv("data/processed/model_ready/X_train_v4.csv", nrows=1)
    train_features = list(x_train.columns)
    selected_features = meta["selected_features"]

    # In v5, 30 features are selected and must match training set features
    assert set(selected_features) == set(train_features)
    assert len(selected_features) == len(train_features)


def test_in_memory_db_models():
    """Verify SQLAlchemy models and relationships in an isolated SQLite memory DB."""
    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)
    TestingSession = sessionmaker(bind=test_engine)
    session = TestingSession()

    req = MaintenanceRequest(
        request_id="REQ-TEST-1",
        corridor_id="CORRIDOR-A",
        department="Engineering",
        maintenance_type="Track Vibration Inspection",
        required_duration_hours=4.0,
        team_size_required=3,
        earliest_start="2026-11-10 08:00:00",
        deadline="2026-11-12 08:00:00",
        priority_score=65.0,
        priority_level="Medium",
    )
    session.add(req)
    session.commit()

    retrieved = session.query(MaintenanceRequest).filter_by(request_id="REQ-TEST-1").first()
    assert retrieved is not None
    assert retrieved.corridor_id == "CORRIDOR-A"
    assert retrieved.priority_score == 65.0
    session.close()
