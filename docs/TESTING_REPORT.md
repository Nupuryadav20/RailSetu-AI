# RailSetu AI — Comprehensive Testing & Hardening Report
_Generated: 2026-11-14 | Test Runner: Pytest 9.2.0 | Python: 3.13.0_

---

## 1. Test Suite Summary

- **Total Test Cases**: 67
- **Passed**: 67 (100%)
- **Failed**: 0
- **Duration**: ~11.8 seconds
- **Test Modules**:
  1. `tests/test_api.py` (24 tests): Core API endpoints, pagination, and query filters.
  2. `tests/test_artifact_service.py` (3 tests): Handling missing artifacts, corrupt CSV/JSON files.
  3. `tests/test_filter_service.py` (1 test): Fallback behavior when filtering missing columns.
  4. `tests/test_model_api.py` (5 tests): Prediction endpoint, empty records, missing features, risk scoring.
  5. `tests/test_v4_api.py` (15 tests): V4 metrics endpoints, threshold analysis, cross-validation, stability, SHAP explanations, image plots, prediction history tracking.
  6. `tests/test_hardening.py` (6 tests): Boundary telemetry handling, monotonic wear scoring, CP-SAT schedule constraints, unscheduled request accounting, SQL-injection safe parameter handling.
  7. `tests/test_v5_api.py` (13 tests): V5 metrics (test/val), model metrics by name, threshold analysis, repeated CV summary, feature selection ranking & ablation, feature distribution stats, database model runs, rich `/ml/predict` diagnostics with z-scores, persistent prediction retrieval, CSV export download, training vs inference feature parity, in-memory SQLite schema validation.

---

## 2. Test Coverage by Component

| Component | Tested Scenarios | Status |
|---|---|---|
| **FastAPI Core & Health** | Root `/`, `/health`, `/api/dashboard/summary` | Passed ✅ |
| **Maintenance APIs** | `/api/maintenance/requests` pagination, filters (department, corridor, priority, text search) | Passed ✅ |
| **Planning & Scheduling** | Weekly/monthly schedules, department breakdowns, corridor allocation | Passed ✅ |
| **Optimization Outputs** | Schedule validation, constraint integrity, request conservation (275 scheduled + 225 unscheduled = 500 total) | Passed ✅ |
| **ML Inference (`/ml/predict`)** | Telemetry ingestion, 30 feature schema verification, thresholding, SHAP contribution generation, z-score computation, SQLite persistence | Passed ✅ |
| **Risk Scoring (`/risk/score`)** | Multi-concept priority scoring, monotonicity with equipment wear | Passed ✅ |
| **V4 Report Serving** | Validation/test metrics, threshold tables, permutation importance, CV summary, plot delivery (PNG binary) | Passed ✅ |
| **V5 ML Endpoints** | Val/Test comparisons, model-specific metrics, feature selection ablation, 5x2 repeated CV, v5 plots | Passed ✅ |
| **Database & Persistence** | Prediction query filtering (class, probability), CSV export streaming, in-memory SQLite schema tests | Passed ✅ |
| **Security & Edge Cases** | Extreme floats, NaN/zero inputs, negative pagination bounds, malicious query strings | Passed ✅ |

---

## 3. Hardening & Fault Tolerance Verification

1. **Pipeline & Model Bundle Resilience**: `load_model_bundle()` returns a full Scikit-Learn `Pipeline([("preprocessor", prep), ("model", model)])` which encapsulates feature transformation, allowing `predict_proba` to work directly on raw pandas DataFrames across all services.
2. **Missing Feature Validation**: HTTP 422 with granular `missing_features` array returned when any of the 30 required telemetry features are absent.
3. **Database Logging**: Real-time predictions are persisted into SQLite (`predictions` table) with structured metadata and SHAP attributions, with zero blocking on user requests.
4. **Distribution Alerts**: Runtime calculation of z-scores warns operators if any physical sensor value deviates by $>3\sigma$ from training distribution.
5. **Feature Parity**: Rigorously verified via `test_feature_parity_training_vs_inference()` that training features, preprocessor columns, and metadata feature lists are strictly identical in name, count, and order.
