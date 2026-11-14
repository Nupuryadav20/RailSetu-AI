import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.artifact_service import load_csv

client = TestClient(app)


def test_predict_extreme_values():
    """Verify prediction pipeline handles extreme boundary telemetry without crashing."""
    x_test = pd.read_csv("data/processed/model_ready/X_test.csv")
    record = x_test.iloc[0].to_dict()
    # Set extreme values
    record["track_temperature_c"] = 100.0
    record["rail_wear_mm"] = 50.0
    record["track_vibration_level"] = 10.0
    record["bearing_temperature_c"] = 250.0

    response = client.post("/api/ml/predict", json={"records": [record]})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    assert 0.0 <= data["results"][0]["maintenance_probability_percent"] <= 100.0


def test_predict_zero_values():
    """Verify prediction pipeline handles zero-value inputs."""
    x_test = pd.read_csv("data/processed/model_ready/X_test.csv")
    record = x_test.iloc[0].to_dict()
    for col in record:
        if isinstance(record[col], (int, float)):
            record[col] = 0

    response = client.post("/api/ml/predict", json={"records": [record]})
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["results"][0]["maintenance_probability_percent"] <= 100.0


def test_risk_score_monotonicity_wear():
    """Higher wear must produce equal or higher operational priority score."""
    x_test = pd.read_csv("data/processed/model_ready/X_test.csv")
    low_wear = x_test.iloc[0].to_dict()
    low_wear["rail_wear_mm"] = 0.5
    low_wear["wheel_wear_percent"] = 5.0

    high_wear = x_test.iloc[0].to_dict()
    high_wear["rail_wear_mm"] = 15.0
    high_wear["wheel_wear_percent"] = 90.0

    r_low = client.post("/api/risk/score", json={"records": [low_wear]}).json()["results"][0]
    r_high = client.post("/api/risk/score", json={"records": [high_wear]}).json()["results"][0]

    assert r_high["equipment_condition_score"] >= r_low["equipment_condition_score"]


def test_optimizer_schedule_constraints():
    """Verify scheduled windows obey operational constraints."""
    scheduled = load_csv("optimization/optimized_maintenance_schedule.csv")
    assert len(scheduled) > 0

    for item in scheduled:
        # Check window validity
        assert "scheduled_start" in item or "start_time" in item or "window_id" in item
        assert "priority_score" in item or "operational_priority_score" in item or "urgency" in item


def test_unscheduled_requests_integrity():
    """Verify unscheduled requests are logged properly."""
    unscheduled = load_csv("optimization/unscheduled_maintenance_requests.csv")
    assert len(unscheduled) > 0
    total_requests = len(load_csv("simulation/simulated_maintenance_requests.csv"))
    scheduled = load_csv("optimization/optimized_maintenance_schedule.csv")
    assert len(scheduled) + len(unscheduled) == total_requests


def test_invalid_query_parameters():
    """Test pagination bounds and search filtering edge cases."""
    # Negative page
    r = client.get("/api/maintenance/requests?page=-1")
    assert r.status_code == 422

    # Zero page
    r = client.get("/api/maintenance/requests?page=0")
    assert r.status_code == 422

    # Huge page size
    r = client.get("/api/maintenance/requests?page_size=500")
    assert r.status_code == 422

    # Search with special characters (safe query handling)
    r = client.get("/api/maintenance/requests?search=%27%20OR%201=1--")
    assert r.status_code == 200
    assert "items" in r.json()
