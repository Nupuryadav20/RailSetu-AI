
import pytest
from pathlib import Path
from fastapi import HTTPException

from backend.app.services import artifact_service


def test_missing_artifact():
    with pytest.raises(HTTPException) as exc:
        artifact_service.get_artifact_path(
            "this_file_does_not_exist.csv"
        )

    assert exc.value.status_code == 404


def test_invalid_csv(monkeypatch, tmp_path):
    fake_file = tmp_path / "invalid.csv"
    fake_file.write_text("invalid csv content")

    monkeypatch.setattr(
        artifact_service,
        "get_artifact_path",
        lambda filename: fake_file
    )

    monkeypatch.setattr(
        artifact_service.pd,
        "read_csv",
        lambda path: (_ for _ in ()).throw(ValueError("CSV error"))
    )

    with pytest.raises(HTTPException) as exc:
        artifact_service.load_dataframe("invalid.csv")

    assert exc.value.status_code == 500


def test_invalid_json(monkeypatch, tmp_path):
    fake_file = tmp_path / "invalid.json"
    fake_file.write_text("{invalid json")

    monkeypatch.setattr(
        artifact_service,
        "get_artifact_path",
        lambda filename: fake_file
    )

    with pytest.raises(HTTPException) as exc:
        artifact_service.load_json("invalid.json")

    assert exc.value.status_code == 500
