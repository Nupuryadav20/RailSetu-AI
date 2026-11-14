# RailSetu AI — Progress Tracker & Master Status
_Last updated: 2026-11-14 | Status: ALL PHASES & FINAL POLISH 100% COMPLETE & VERIFIED_

---

## Overall Status Summary

All tasks of the **V5 Upgrade & Final Polish Pass** are complete and fully verified.
- **Backend Tests:** 67/67 tests passing (`./.venv/bin/python -m pytest tests/`)
- **Frontend Build:** Passing cleanly (`npm run build --prefix frontend`, built in ~130ms, 0 errors)
- **Active Model Served:** V5 XGBoost Pipeline with 30 selected features, Isotonic Calibration (Test Brier 0.1474), and threshold 0.45
- **Database Layer:** SQLAlchemy 2 + SQLite (`data/railsetu.db`), seeded via `python -m scripts.seed_db`
- **Design System & CSS Audit:** Zero unstyled BEM classes. Enterprise Light Theme with responsive sidebar drawer.
- **Visual Evidence:** All 7 pages screenshotted at 1440px and 390px in `docs/screenshots/` (14 total PNG images).
- **System Verification:** All 18 API endpoints verified (100% PASS), audited in `docs/VERIFICATION.md`.
- **Zero Fake Numbers:** All metrics, thresholds, CV scores, feature stats, and ablation numbers derived from real training/evaluation runs.

---

## Detailed Task Verification Matrix

| Task | Status | Evidence & Verification |
|---|---|---|
| **TASK 1: Model Improvement (v5)** | ✅ COMPLETE | `backend/ml/train_v5.py` executed. Evaluated 10 models (Dummy, LogReg, DT, RF, ExtraTrees, HistGB, XGBoost, LightGBM, CatBoost, Voting Ensemble). Optuna tuned LightGBM. Val PR-AUC: XGBoost 0.33202 vs v4 0.33026 (+0.00176 margin, within CV std 0.00548). Test PR-AUC: 0.32955. Isotonic calibration cut Test Brier from 0.2259 to 0.1474. MLflow experiment `railsetu_v5`. Artifacts in `backend/artifacts/models/v5/` and `reports/v5/`. |
| **TASK 2: Feature Selection** | ✅ COMPLETE | Ablation executed across subsets: Top 5 (Val PR-AUC 0.2052), Top 8 (0.2464), Top 10 (0.2828), Top 12 (0.2974), Top 15 (0.3150), Top 20 (0.3253), Top 30 (0.3320). 1-std boundary preserves 30 features. Signal ceiling diagnosed: low mutual info (<0.01) and depth 2-3 tree plateau. |
| **TASK 3: Database Layer** | ✅ COMPLETE | SQLAlchemy 2 models in `backend/app/database/models.py`. Connection manager in `backend/app/database/connection.py`. Database seeder `scripts/seed_db.py` populates 500 requests, 140 block windows, 275 scheduled assignments, 225 unscheduled requests, 2 model runs, 990 metrics, 20 threshold evaluations, 50 feature importances, 30 feature distribution stats. Complete ER diagram in `docs/DATABASE.md`. |
| **TASK 4: API Additions** | ✅ COMPLETE | Added `/api/ml/v5/comparison`, `/api/ml/v5/metrics/{model}`, `/api/ml/v5/thresholds`, `/api/ml/v5/cv`, `/api/ml/v5/feature-selection`, `/api/ml/v5/feature-stats`, `/api/ml/v5/experiments`, `/api/ml/v5/metadata`, `/api/ml/plots/v5/{name}`, `/api/ml/predictions`, `/api/ml/predictions/download`. Upgraded `/api/ml/predict` to calculate z-scores, percentiles, distribution alerts, SHAP attributions, and store predictions into the database. |
| **TASK 5: Frontend Upgrade** | ✅ COMPLETE | React 19 + TypeScript + Vite + Tailwind CSS. `ModelInsights.tsx` upgraded with V5 vs V4 selector, full 10-model comparison matrix, feature ablation table, repeated CV metrics, threshold curve, and MLflow run history. `PredictiveMaintenance.tsx` upgraded with 30-feature real-time distribution diagnostics table, calibrated probability, out-of-distribution alerts, persistent database history, and CSV download. Clean build: 0 errors. |
| **TASK 6: Final Polish & Verification** | ✅ COMPLETE | Root-caused CSS glitch on Maintenance page (missing BEM classes in index.css), rebuilt with 3 KPI cards, responsive filter card, sortable table, risk mini-bar, pagination, and details drawer. Fixed 0 unstyled classes across all 7 pages. Added Model in Production card, 10-model ranking table with $\le 1\sigma$ tie detection, selection rules explanation (English + Hinglish), and verdict card. Generated 14 screenshots (1440px desktop & 390px mobile). Ran full end-to-end verification report in `docs/VERIFICATION.md`. |

---

## 3-Concept Architecture Separation

1. **Concept A (ML Failure Probability):** XGBoost statistical estimation based strictly on 30 telemetry and operational features. Calibrated with Isotonic regression. Kept independent of labels and risk formulas.
2. **Concept B (Dataset Risk Score):** Pre-existing synthetic column excluded from feature matrix to prevent leakage.
3. **Concept C (Operational Priority Score):** Rule-based composite score weighting ML probability (40%) and physical equipment condition metrics (60%) for track and rolling stock scheduling.
