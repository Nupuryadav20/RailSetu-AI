"""
P3 — Update /ml/predict endpoint + add new API endpoints for reports/plots.
Run AFTER train_v4.py completes and v4_metadata.json is confirmed.
"""
# This file documents planned API changes; actual edits in dashboard.py and model_service.py
# See backend/app/api/dashboard.py and backend/app/services/model_service.py

PLAN = """
CHANGES NEEDED IN model_service.py:
1. load_model_bundle() — add version selection (v4 if available, else v1)
2. predict_records() — add: model_version, timestamp, chosen_threshold, top_shap_contributors, is_calibrated note
3. Keep old v1 path working (don't break existing tests)

NEW ENDPOINTS IN dashboard.py:
- GET /api/ml/metrics/full      → serves reports/v4/master_metrics.csv
- GET /api/ml/metrics/val       → reports/v4/val_model_comparison.csv  
- GET /api/ml/metrics/test      → reports/v4/test_model_comparison.csv
- GET /api/ml/threshold-analysis → reports/v4/threshold_analysis.csv
- GET /api/ml/cv-summary        → reports/v4/cv_summary.json
- GET /api/ml/seed-stability    → reports/v4/seed_stability.csv
- GET /api/ml/permutation-importance → reports/v4/permutation_importance.csv
- GET /api/ml/shap/v4           → reports/v4/shap_feature_importance_v4.csv
- GET /api/ml/shap/examples     → reports/v4/shap_worked_examples.json
- GET /api/ml/plots/{name}      → serves PNG from reports/v4/{name}.png
- GET /api/ml/metadata          → reports/v4/v4_metadata.json
- GET /api/predict/history      → in-memory list (last 50, no sensitive data)
- GET /api/dataset/stats        → computed from railway_cleaned.csv
"""
