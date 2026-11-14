import pytest
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


@pytest.mark.parametrize("endpoint", [
    "/api/ml/metrics/full",
    "/api/ml/metrics/val",
    "/api/ml/metrics/test",
    "/api/ml/threshold-analysis",
    "/api/ml/cv-summary",
    "/api/ml/seed-stability",
    "/api/ml/permutation-importance",
    "/api/ml/shap/v4",
    "/api/ml/shap/examples",
    "/api/ml/metadata",
    "/api/predict/history",
    "/api/dataset/stats",
])
def test_v4_json_endpoints(endpoint):
    response = client.get(endpoint)
    assert response.status_code == 200, f"{endpoint} returned {response.status_code}"
    assert response.headers["content-type"].startswith("application/json")
    data = response.json()
    assert isinstance(data, (dict, list))


def test_v4_plot_endpoint():
    response = client.get("/api/ml/plots/01_class_distribution")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 1000


def test_predict_includes_version_and_metadata():
    x_test = pd.read_csv("data/processed/model_ready/X_test.csv")
    payload = {"records": [x_test.iloc[0].to_dict()]}
    response = client.post("/api/ml/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "timestamp" in data
    assert "honest_note" in data
    assert "concept_note" in data
    assert "results" in data
    assert len(data["results"]) == 1
    assert "decision" in data["results"][0]

    # Check history endpoint updated
    hist_response = client.get("/api/predict/history")
    assert hist_response.status_code == 200
    hist = hist_response.json()
    assert len(hist) >= 1
    assert "probability_percent" in hist[-1]


def test_dataset_stats_content():
    response = client.get("/api/dataset/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 100000
    assert data["ml_features_count"] == 30
    assert data["is_live_railway_data"] is False
    assert "risk_score" in data["leakage_excluded_columns"]
