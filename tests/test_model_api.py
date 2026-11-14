
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_prediction_success():
    x_test = pd.read_csv("data/processed/model_ready/X_test.csv")
    payload = {"records": [x_test.iloc[0].to_dict()]}

    response = client.post("/api/ml/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["model"] == "XGBoost"
    assert len(data["results"]) == 1
    assert 0 <= data["results"][0]["maintenance_probability_percent"] <= 100
    assert data["results"][0]["predicted_class"] in [0, 1]


def test_prediction_missing_features():
    response = client.post(
    "/api/ml/predict",
    json={"records": [{"region": "Southern Railway"}]}
)

    assert response.status_code == 422


def test_prediction_empty_records():
    response = client.post(
        "/api/ml/predict",
        json={"records": []}
    )

    assert response.status_code == 422


def test_risk_scoring_success():
    x_test = pd.read_csv("data/processed/model_ready/X_test.csv")
    payload = {"records": [x_test.iloc[0].to_dict()]}
    response = client.post("/api/risk/score", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["model"] == "XGBoost"
    assert len(data["results"]) == 1

    result = data["results"][0]
    assert 0 <= result["operational_priority_score"] <= 100
    assert result["priority_level"] in ["High", "Medium", "Low"]
    assert "disclaimer" in data


def test_risk_scoring_missing_features():
    response = client.post(
    "/api/risk/score",
    json={"records": [{"region": "Southern Railway"}]}
)

    assert response.status_code == 422
