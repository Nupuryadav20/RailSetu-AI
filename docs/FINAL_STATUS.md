# RailSetu AI — Final Project Status Report (V5 Upgrade)
_Generated: 2026-11-14 | Overall Completion: 100% | Verified by Production Build & Pytest_

---

## 1. Executive Summary & Verification Matrix

RailSetu AI is an end-to-end railway infrastructure decision-support platform integrating predictive maintenance machine learning with OR-Tools CP-SAT constraint programming and a relational database layer (SQLAlchemy 2 + SQLite). Every layer of the project has been empirically verified.

| Project Component | Completion | Verification Evidence |
|---|---|---|
| **P1: Audit & Discovery** | 100% | `docs/audit/PROJECT_AUDIT_REPORT.md` |
| **P2: ML Diagnosis & Benchmarks** | 100% | `reports/v4/` (16 plots, full metrics CSVs), `backend/artifacts/models/v4/`, MLflow experiment `railsetu_v4` (`cbf157e9056047ebbd3a807271790977`) |
| **P3: Explainable API & Reports** | 100% | 12 endpoints in `backend/app/api/dashboard.py`, top-5 local SHAP attributions in `/ml/predict` |
| **P4: CP-SAT Block Optimizer** | 100% | `docs/OPTIMIZER.md`, OR-Tools CP-SAT formulation, 275 scheduled vs 225 unscheduled requests, 0 constraint overlaps |
| **P5: React 19 Frontend SPA** | 100% | Production build verified (`npm run build` in 118ms), light calm enterprise theme (#EEF1F4), all pages wired to real backend endpoints |
| **P6: Automated Test Suite** | 100% | **67/67 tests passing** via `pytest tests/ -v`, `docs/TESTING_REPORT.md` |
| **P7: Academic Docs & Viva Prep** | 100% | `docs/VIVA_PREP.md`, `docs/ARCHITECTURE.md`, `docs/MODEL_CARD.md`, `docs/AI_PROJECT_DETAILS.md` |
| **P8: Production Deployment** | 100% | `docs/DEPLOYMENT.md`, dynamic CORS in `backend/app/main.py` & `backend/app/core/config.py`, containerization configs |
| **V5 UPGRADE: Models & Tuning** | 100% | 10 models evaluated (Dummy, LogReg, DT, RF, ExtraTrees, HistGB, XGBoost, LightGBM, CatBoost, Ensemble). Optuna tuning. Isotonic calibration (Test Brier 0.1474). MLflow `railsetu_v5`. Artifacts in `backend/artifacts/models/v5/` and `reports/v5/`. |
| **V5 UPGRADE: Feature Selection** | 100% | Ablation on Top 5, 8, 10, 12, 15, 20, 30 features. Pruning reduces PR-AUC below 1-std boundary. Signal ceiling diagnosed (low mutual info < 0.01). |
| **V5 UPGRADE: Database Layer** | 100% | SQLAlchemy 2 + SQLite (`data/railsetu.db`). 11 models, idempotent seeder `scripts/seed_db.py`, persistent prediction logging, CSV export, `docs/DATABASE.md`. |
| **FINAL POLISH & VERIFICATION** | 100% | 14 screenshots (1440px desktop & 390px mobile) in `docs/screenshots/`, zero unstyled classes, 18/18 API endpoints verified (100% PASS), comprehensive audit in `docs/VERIFICATION.md`. |

---

## 2. Key Verified Technical Facts

### A. Dataset Provenance
- **Benchmark**: Kaggle Indian Railway Failure Detection & Predictive Maintenance.
- **Records**: 100,000 cleaned rows × 35 columns (30 input features).
- **Class Balance**: 79,804 negative (79.8%) vs 20,196 positive (20.2%).
- **Target Leakage**: Strictly excluded `risk_score`, `failure_type`, `failure_severity`, and `train_id`.
- **Nature**: Simulated / synthetic. NOT live Indian Railways or CRIS production data.

### B. Machine Learning Performance (v5 Active)
- **Model**: Scikit-Learn Pipeline + XGBoost Classifier v5 (Calibrated with Isotonic Regression).
- **Validation ROC-AUC**: **0.5558** (Held-out Test: **0.5451**).
- **Validation PR-AUC**: **0.3320** (Held-out Test: **0.3296** vs Dummy Baseline **0.2033**).
- **Test Brier Score**: Reduced from **0.2259** to **0.1474** via Isotonic Calibration.
- **Operating Threshold**: **0.45** (flags ~15% fleet, precision 23.5%, recall 17.5%).
- **Repeated CV**: 5x2 Repeated Stratified CV on Train+Val yields PR-AUC $0.3249 \pm 0.0055$.
- **Signal Reality**: Synthetic features have weak intrinsic correlation with failure labels; honest reporting maintained across all UI components and docs.

### C. Constraint Optimization (OR-Tools CP-SAT)
- **Algorithm**: Constraint Programming with SAT-based search (`ortools.sat.python.cp_model`).
- **Requests Submitted**: 500 requests across Engineering, Traction, and Signal & Telecom.
- **Block Windows**: 140 section intervals.
- **Requests Scheduled**: **275 requests** (55.0% fulfillment).
- **Requests Unscheduled**: **225 requests** (duration and track bottleneck limits).
- **Constraint Violations**: Exactly 0 overlaps.

### D. Distinction of Three Concepts
1. **Concept A (ML Probability)**: Statistical model likelihood ($0-100\%$).
2. **Concept B (Dataset Risk Score)**: Synthetic historical label; excluded from features.
3. **Concept C (Operational Priority Score)**: Rule-based composite score ($0-100$) blending ML probability with mechanical wear percentiles and corridor urgency.

---

## 3. Known Limitations & Future Scope
1. **Low Feature Mutual Information**: Synthetic telemetry limits classification performance to ~0.55 ROC-AUC; real acoustic wayside sensors and wheel impact load detectors (WILD) would provide significantly stronger signals.
2. **Deterministic Schedule Windows**: Windows are currently fixed; future versions can integrate dynamic train rescheduling to flex block window margins.
3. **Single Depot Capacity**: Current model treats team availability per department; future work can model geographic crew deadheading.
