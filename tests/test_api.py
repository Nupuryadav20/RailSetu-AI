
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


@pytest.mark.parametrize("endpoint", [
    "/",
    "/health",
    "/api/dashboard/summary",
    "/api/maintenance/requests",
    "/api/maintenance/priority",
    "/api/schedule/optimized",
    "/api/schedule/unscheduled",
    "/api/planning/weekly",
    "/api/planning/monthly",
    "/api/planning/departments",
    "/api/planning/corridors",
    "/api/ml/model-comparison",
    "/api/ml/best-model",
    "/api/ml/feature-importance",
    "/api/ml/prediction-explanations",
    "/api/optimization/validation",
    "/api/simulation/block-windows",
])
def test_api_endpoints(endpoint):
    response = client.get(endpoint)

    assert response.status_code == 200, (
        f"{endpoint} failed: {response.status_code}"
    )

    assert response.headers["content-type"].startswith(
        "application/json"
    )

    assert isinstance(response.json(), (dict, list))

def test_maintenance_pagination():
    response = client.get(
        "/api/maintenance/requests?page=1&page_size=10"
    )
    assert response.status_code == 200

    data = response.json()
    assert len(data["items"]) == 10
    assert data["pagination"]["page"] == 1
    assert data["pagination"]["page_size"] == 10


def test_department_filter():
    response = client.get(
        "/api/maintenance/requests?department=Engineering"
    )
    assert response.status_code == 200

    items = response.json()["items"]
    assert len(items) > 0
    assert all(
        item["department"].lower() == "engineering"
        for item in items
    )


def test_priority_filter():
    response = client.get(
        "/api/maintenance/requests?priority=Medium"
    )
    assert response.status_code == 200

    items = response.json()["items"]
    assert all(
        item["priority_level"].lower() == "medium"
        for item in items
    )


def test_corridor_filter():
    response = client.get(
        "/api/maintenance/requests?corridor=CORRIDOR-0"
    )
    assert response.status_code == 200

    items = response.json()["items"]
    assert all(
        item["corridor_id"].lower() == "corridor-0"
        for item in items
    )


def test_search_filter():
    response = client.get(
        "/api/maintenance/requests?search=Engineering"
    )
    assert response.status_code == 200

    items = response.json()["items"]
    assert len(items) > 0


def test_combined_filters():
    response = client.get(
        "/api/maintenance/requests"
        "?department=Engineering&priority=Medium"
    )
    assert response.status_code == 200

    items = response.json()["items"]
    assert all(
        item["department"].lower() == "engineering"
        and item["priority_level"].lower() == "medium"
        for item in items
    )


def test_invalid_page_size():
    response = client.get(
        "/api/maintenance/requests?page_size=101"
    )
    assert response.status_code == 422
