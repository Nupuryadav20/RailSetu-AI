import json
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from backend.app.database.connection import engine, init_db, SessionLocal
from backend.app.database.models import (
    MaintenanceRequest,
    BlockWindow,
    ScheduleAssignment,
    UnscheduledRequest,
    Prediction,
    ModelRun,
    ModelMetric,
    ThresholdMetric,
    FeatureImportance,
    FeatureStat,
    DatasetInfo,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def seed_all():
    print("Initializing Database schema...")
    init_db()

    session: Session = SessionLocal()
    try:
        # Clear existing seeded tables (idempotent)
        session.query(MaintenanceRequest).delete()
        session.query(BlockWindow).delete()
        session.query(ScheduleAssignment).delete()
        session.query(UnscheduledRequest).delete()
        session.query(ModelRun).delete()
        session.query(ModelMetric).delete()
        session.query(ThresholdMetric).delete()
        session.query(FeatureImportance).delete()
        session.query(FeatureStat).delete()
        session.query(DatasetInfo).delete()
        session.commit()

        print("1. Seeding MaintenanceRequests...")
        maint_csv = REPO_ROOT / "backend/artifacts/simulation/simulated_maintenance_requests.csv"
        if maint_csv.exists():
            df_maint = pd.read_csv(maint_csv)
            maint_objs = []
            for _, r in df_maint.iterrows():
                maint_objs.append(
                    MaintenanceRequest(
                        request_id=str(r.get("request_id", "")),
                        corridor_id=str(r.get("corridor_id", "")),
                        department=str(r.get("department", "")),
                        maintenance_type=str(r.get("maintenance_type", "")),
                        required_duration_hours=float(r.get("required_duration_hours", 0.0)),
                        team_size_required=int(r.get("team_size_required", 1)),
                        earliest_start=str(r.get("earliest_start", "")),
                        deadline=str(r.get("deadline", "")),
                        priority_score=float(r.get("operational_priority_score", 0.0)),
                        priority_level=str(r.get("priority_level", "Medium")),
                    )
                )
            session.bulk_save_objects(maint_objs)
            print(f"   Seeded {len(maint_objs)} MaintenanceRequests.")

        print("2. Seeding BlockWindows...")
        win_csv = REPO_ROOT / "backend/artifacts/simulation/simulated_block_windows.csv"
        if win_csv.exists():
            df_win = pd.read_csv(win_csv)
            win_objs = []
            for _, r in df_win.iterrows():
                win_objs.append(
                    BlockWindow(
                        window_id=str(r.get("window_id", "")),
                        corridor_id=str(r.get("corridor_id", "")),
                        window_start=str(r.get("window_start", "")),
                        window_end=str(r.get("window_end", "")),
                        window_status=str(r.get("window_status", "Available")),
                        max_parallel_teams=int(r.get("max_parallel_teams", 1)),
                    )
                )
            session.bulk_save_objects(win_objs)
            print(f"   Seeded {len(win_objs)} BlockWindows.")

        print("3. Seeding ScheduleAssignments...")
        sch_csv = REPO_ROOT / "backend/artifacts/optimization/optimized_maintenance_schedule.csv"
        if sch_csv.exists():
            df_sch = pd.read_csv(sch_csv)
            sch_objs = []
            for _, r in df_sch.iterrows():
                sch_objs.append(
                    ScheduleAssignment(
                        request_id=str(r.get("request_id", "")),
                        window_id=str(r.get("window_id", "")),
                        corridor_id=str(r.get("corridor_id", "")),
                        department=str(r.get("department", "")),
                        scheduled_start=str(r.get("scheduled_start", "")),
                        scheduled_end=str(r.get("scheduled_end", "")),
                        priority_score=float(r.get("priority_score", 0.0)),
                    )
                )
            session.bulk_save_objects(sch_objs)
            print(f"   Seeded {len(sch_objs)} ScheduleAssignments.")

        print("4. Seeding UnscheduledRequests...")
        unsch_csv = REPO_ROOT / "backend/artifacts/optimization/unscheduled_maintenance_requests.csv"
        if unsch_csv.exists():
            df_unsch = pd.read_csv(unsch_csv)
            unsch_objs = []
            for _, r in df_unsch.iterrows():
                priority = r.get("final_priority_score", r.get("operational_priority_score", 0.0))
                priority_lvl = r.get("final_priority_level", r.get("priority_level", "Medium"))
                unsch_objs.append(
                    UnscheduledRequest(
                        request_id=str(r.get("request_id", "")),
                        corridor_id=str(r.get("corridor_id", "")),
                        department=str(r.get("department", "")),
                        priority_score=float(priority) if pd.notnull(priority) else 0.0,
                        priority_level=str(priority_lvl),
                        reason=str(r.get("unscheduled_reason", "")),
                    )
                )
            session.bulk_save_objects(unsch_objs)
            print(f"   Seeded {len(unsch_objs)} UnscheduledRequests.")

        print("5. Seeding ModelRuns...")
        v5_meta_path = REPO_ROOT / "reports/v5/v5_metadata.json"
        v4_meta_path = REPO_ROOT / "reports/v4/v4_metadata.json"
        v5_meta = {}
        if v5_meta_path.exists():
            with open(v5_meta_path) as f:
                v5_meta = json.load(f)
        v4_meta = {}
        if v4_meta_path.exists():
            with open(v4_meta_path) as f:
                v4_meta = json.load(f)

        runs = [
            ModelRun(
                mlflow_run_id=v4_meta.get("mlflow_run_id", "v4_run"),
                model_name=v4_meta.get("best_model_name", "RandomForest"),
                version="v4",
                params_json=json.dumps(v4_meta.get("model_parameters", {})),
                created_at=v4_meta.get("training_timestamp", "2026-11-14"),
                is_served=False,
            ),
            ModelRun(
                mlflow_run_id=v5_meta.get("mlflow_run_id", "v5_run"),
                model_name=v5_meta.get("best_model_name", "XGBoost"),
                version="v5",
                params_json=json.dumps(v5_meta.get("best_params", {})),
                created_at=v5_meta.get("trained_at", "2026-11-14"),
                is_served=True,
            ),
        ]
        session.bulk_save_objects(runs)
        print(f"   Seeded {len(runs)} ModelRuns.")

        print("6. Seeding ModelMetrics (v4 and v5)...")
        metric_objs = []
        for v_tag, csv_file in [
            ("v4", REPO_ROOT / "reports/v4/master_metrics.csv"),
            ("v5", REPO_ROOT / "reports/v5/master_metrics_v5.csv"),
        ]:
            if csv_file.exists():
                df_m = pd.read_csv(csv_file)
                metric_cols = [c for c in df_m.columns if c not in ["model", "split"]]
                for _, r in df_m.iterrows():
                    m_name = str(r["model"])
                    s_name = str(r["split"])
                    for mc in metric_cols:
                        val = r[mc]
                        if pd.notnull(val):
                            metric_objs.append(
                                ModelMetric(
                                    model_name=m_name,
                                    version=v_tag,
                                    split=s_name,
                                    metric=mc,
                                    value=float(val),
                                )
                            )
        session.bulk_save_objects(metric_objs)
        print(f"   Seeded {len(metric_objs)} ModelMetrics.")

        print("7. Seeding ThresholdMetrics...")
        thresh_objs = []
        for v_tag, csv_file in [
            ("v4", REPO_ROOT / "reports/v4/threshold_analysis.csv"),
            ("v5", REPO_ROOT / "reports/v5/threshold_analysis_v5.csv"),
        ]:
            if csv_file.exists():
                df_th = pd.read_csv(csv_file)
                for _, r in df_th.iterrows():
                    thresh_objs.append(
                        ThresholdMetric(
                            version=v_tag,
                            threshold=float(r["threshold"]),
                            precision=float(r["precision"]),
                            recall=float(r["recall"]),
                            f1=float(r["f1"]),
                            f2=float(r["f2"]),
                            tp=int(r["tp"]),
                            fp=int(r["fp"]),
                            tn=int(r["tn"]),
                            fn=int(r["fn"]),
                            pct_flagged=float(r["pct_flagged"]),
                            operational_character=str(r.get("operational_character", "")),
                        )
                    )
        session.bulk_save_objects(thresh_objs)
        print(f"   Seeded {len(thresh_objs)} ThresholdMetrics.")

        print("8. Seeding FeatureImportance...")
        feat_objs = []
        # v4
        v4_fi_path = REPO_ROOT / "reports/v4/feature_importance.csv"
        v4_pi_path = REPO_ROOT / "reports/v4/permutation_importance.csv"
        v4_shap_path = REPO_ROOT / "reports/v4/shap_feature_importance_v4.csv"
        v4_fi = pd.read_csv(v4_fi_path).set_index("feature") if v4_fi_path.exists() else pd.DataFrame()
        v4_pi = pd.read_csv(v4_pi_path).set_index("feature") if v4_pi_path.exists() else pd.DataFrame()
        v4_shap = pd.read_csv(v4_shap_path).set_index("feature") if v4_shap_path.exists() else pd.DataFrame()

        all_v4_feats = list(v4_fi.index) if len(v4_fi) > 0 else []
        for f in all_v4_feats:
            f_tree = float(v4_fi.loc[f, "importance"]) if f in v4_fi.index else 0.0
            f_perm = float(v4_pi.loc[f, "importance_mean"]) if f in v4_pi.index else 0.0
            f_shap = float(v4_shap.loc[f, "mean_abs_shap"]) if f in v4_shap.index else 0.0
            feat_objs.append(
                FeatureImportance(
                    version="v4",
                    feature=f,
                    importance_perm=f_perm,
                    importance_tree=f_tree,
                    mean_abs_shap=f_shap,
                )
            )

        # v5
        v5_fi_path = REPO_ROOT / "reports/v5/feature_ranking_v5.csv"
        if v5_fi_path.exists():
            df_v5_fi = pd.read_csv(v5_fi_path)
            for _, r in df_v5_fi.iterrows():
                feat_objs.append(
                    FeatureImportance(
                        version="v5",
                        feature=str(r["feature"]),
                        importance_perm=float(r.get("importance_perm", 0.0)),
                        importance_tree=float(r.get("importance_tree", 0.0)),
                        mean_abs_shap=float(r.get("mean_abs_shap", 0.0)),
                    )
                )
        session.bulk_save_objects(feat_objs)
        print(f"   Seeded {len(feat_objs)} FeatureImportance rows.")

        print("9. Calculating and Seeding FeatureStat...")
        x_tr_path = REPO_ROOT / "data/processed/model_ready/X_train_v4.csv"
        y_tr_path = REPO_ROOT / "data/processed/model_ready/y_train_v4.csv"
        if x_tr_path.exists() and y_tr_path.exists():
            X_train = pd.read_csv(x_tr_path)
            y_train = pd.read_csv(y_tr_path).squeeze()
            stat_objs = []
            for col in X_train.columns:
                series = X_train[col]
                is_num = pd.api.types.is_numeric_dtype(series)
                if is_num:
                    s_clean = series.dropna()
                    c0_series = s_clean[y_train == 0]
                    c1_series = s_clean[y_train == 1]
                    stat_objs.append(
                        FeatureStat(
                            feature=col,
                            data_type="numeric",
                            min_val=float(s_clean.min()),
                            max_val=float(s_clean.max()),
                            mean_val=float(s_clean.mean()),
                            std_val=float(s_clean.std()),
                            p25=float(s_clean.quantile(0.25)),
                            p50=float(s_clean.quantile(0.50)),
                            p75=float(s_clean.quantile(0.75)),
                            class_0_mean=float(c0_series.mean()) if len(c0_series) > 0 else 0.0,
                            class_1_mean=float(c1_series.mean()) if len(c1_series) > 0 else 0.0,
                            categories_json=json.dumps([]),
                        )
                    )
                else:
                    cats = list(series.dropna().unique())
                    stat_objs.append(
                        FeatureStat(
                            feature=col,
                            data_type="categorical",
                            min_val=0.0,
                            max_val=0.0,
                            mean_val=0.0,
                            std_val=0.0,
                            p25=0.0,
                            p50=0.0,
                            p75=0.0,
                            class_0_mean=0.0,
                            class_1_mean=0.0,
                            categories_json=json.dumps(cats),
                        )
                    )
            session.bulk_save_objects(stat_objs)
            print(f"   Seeded {len(stat_objs)} FeatureStats.")

        print("10. Seeding DatasetInfo...")
        info_items = [
            DatasetInfo(
                key="metadata",
                value_json=json.dumps(
                    {
                        "total_rows": 50000,
                        "raw_columns": 33,
                        "model_features": 30,
                        "target_column": "maintenance_required",
                        "leakage_columns_excluded": ["risk_score", "failure_type", "failure_severity"],
                        "train_val_test_split": {"train": 30000, "val": 10000, "test": 10000},
                        "target_distribution": {"class_0": 39990, "class_1": 10010, "positive_rate_percent": 20.02},
                        "synthetic_disclosure": "Dataset is purely synthetic for research and simulation benchmarking. Not calibrated or certified for live railway signaling/safety operations.",
                    }
                ),
            )
        ]
        session.bulk_save_objects(info_items)
        print("   Seeded DatasetInfo.")

        session.commit()
        print("Database seeded successfully!")
    finally:
        session.close()


if __name__ == "__main__":
    seed_all()
