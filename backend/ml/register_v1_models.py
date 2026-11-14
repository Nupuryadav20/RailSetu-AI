import json
import math
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
BACKUP_DIR = ROOT / "backend/artifacts/backup/v1"
DATA_DIR = ROOT / "data/processed/model_ready"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_registry_uri("http://127.0.0.1:5000")

EXPERIMENT_NAME = "RailSetu-V1-Registry"
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient()

models = {
    "Logistic Regression": {
        "file": "logistic_regression.joblib",
        "registry": "RailSetu_Logistic_Regression",
    },
    "Random Forest": {
        "file": "random_forest.joblib",
        "registry": "RailSetu_Random_Forest",
    },
    "XGBoost": {
        "file": "xgboost.joblib",
        "registry": "RailSetu_XGBoost",
    },
}

comparison = pd.read_csv(BACKUP_DIR / "model_comparison.csv")
thresholds = json.loads(
    (BACKUP_DIR / "model_thresholds.json").read_text()
)
best_model = json.loads(
    (BACKUP_DIR / "best_model.json").read_text()
)

X_sample = pd.read_csv(DATA_DIR / "X_train.csv").head(2)

print("\nStarting safe V1 registration...\n")

for display_name, config in models.items():

    registry_name = config["registry"]

    # Never create duplicate registrations
    existing = client.search_model_versions(
        f"name='{registry_name}'"
    )

    if existing:
        print(f"SKIPPED: {registry_name} already exists.")
        continue

    print(f"\nProcessing: {display_name}")

    model_path = BACKUP_DIR / config["file"]
    model = joblib.load(model_path)
    model.predict(X_sample)

    row = comparison.loc[
        comparison["model"] == display_name
    ].iloc[0]

    threshold_info = thresholds[display_name]
    threshold = float(threshold_info["threshold"])

    with mlflow.start_run(run_name=f"V1_{display_name}") as run:

        mlflow.set_tags({
            "project": "RailSetu AI",
            "model_version": "V1",
            "dataset_type": "synthetic_simulated",
            "registration_type": "existing_model",
            "threshold_status": threshold_info["status"],
            "selected_best_model": str(
                best_model["best_model"] == display_name
            ),
        })

        mlflow.log_params({
            "model_name": display_name,
            "threshold": threshold,
            "selection_metric": best_model["selection_metric"],
            "dataset": "Indian Railway Failure Detection and Maintenance 100K",
        })

        # Log evaluation metrics
        metrics = {}

        for key, value in row.items():
            if key in ("model", "threshold_status"):
                continue

            numeric_value = pd.to_numeric(value, errors="coerce")

            if pd.notna(numeric_value) and math.isfinite(float(numeric_value)):
                metrics[key] = float(numeric_value)

        mlflow.log_metrics(metrics)

        # Preserve V1 metadata
        for filename in [
            "model_comparison.csv",
            "model_thresholds.json",
            "best_model.json",
            "train_model_v1.py",
        ]:
            mlflow.log_artifact(str(BACKUP_DIR / filename))

        # Register the existing fitted model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_sample,
            registered_model_name=registry_name,
            serialization_format="cloudpickle",
        )

        run_id = run.info.run_id

    # Find the version created by this run
    versions = client.search_model_versions(
        f"name='{registry_name}'"
    )

    matching = [
        version for version in versions
        if version.run_id == run_id
    ]

    if not matching:
        raise RuntimeError(
            f"Registration version not found for {registry_name}"
        )

    latest = matching[0]

    client.set_model_version_tag(
        registry_name, latest.version, "project_version", "V1"
    )
    client.set_model_version_tag(
        registry_name,
        latest.version,
        "threshold_status",
        threshold_info["status"],
    )

    print(
        f"REGISTERED: {registry_name} | Version: {latest.version}"
    )

print("\nV1 registration process completed.")
print("Open: http://127.0.0.1:5000")
