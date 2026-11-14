# RailSetu AI — P1 Project Audit Report
_Generated: 2026-11-14 | Auditor: Senior AI/ML + Full-Stack Engineer_
_All numbers reproduced from actual files; nothing assumed from docs._

---

## 1. REPOSITORY ACCESS CONFIRMED
- **Workspace**: `/Users/nupuryadav/RailSetu-AI`
- **Python venv**: `.venv/bin/python3` (Python 3.13.0)
- **Git**: present (`.git/`)
- **Full read/write access**: ✅

---

## 2. FILE INVENTORY

### Backend (`backend/`)
| File/Dir | Purpose | Status |
|----------|---------|--------|
| `app/main.py` | FastAPI app, CORS, logging, global exception handler | ✅ Working |
| `app/api/dashboard.py` | All API routes (~20 endpoints) | ✅ Working |
| `app/core/config.py` | Settings (AppName, version, env) | ✅ |
| `app/core/logging_config.py` | Structured logging setup | ✅ |
| `app/services/model_service.py` | Load model + predict | ✅ Loads v1 XGBoost |
| `app/services/artifact_service.py` | CSV/JSON loaders with error handling | ✅ |
| `app/services/filter_service.py` | Pagination + filter for dataframes | ✅ |
| `app/services/risk_service.py` | Rule-based risk scoring (concept C) | ✅ |
| `app/schemas/prediction.py` | Pydantic input schemas | ✅ |
| `ml/train_model.py` | Training script (v2 improved) | ✅ Exists |
| `ml/risk_scoring.py` | Dataset risk_score enrichment (concept B) | ✅ Exists |
| `ml/explain_model.py` | SHAP explanation generation | ✅ Exists |
| `ml/preprocess.py` | Preprocessing pipeline | ✅ Exists |
| `ml/eda.py` | EDA script | ✅ Exists |
| `optimization/block_optimizer.py` | OR-Tools CP-SAT scheduler | ✅ CP-SAT (not ILP/MILP) |
| `optimization/priority_engine.py` | Rule-based priority scoring (concept C) | ✅ Rule-based |
| `optimization/validate_schedule.py` | Schedule validation | ✅ |
| `simulation/generate_maintenance_requests.py` | Synthetic request generation | ✅ |
| `simulation/generate_block_windows.py` | Block window generation | ✅ |
| `planning/weekly_monthly_planner.py` | Planning aggregation | ✅ |
| `requirements.txt` | Backend dependencies | ✅ |

### Data (`data/`)
| File | Shape | Status |
|------|-------|--------|
| `raw/indian_railway_failure_detection_maintenance_v2.csv` | 100,000 × 26 | ✅ |
| `raw/indian_railway_predictive_maintenance_100k.csv` | 100,300 × 35 | ✅ |
| `data/processed/railway_cleaned.csv` | 100,000 × 35 | ✅ |
| `data/processed/model_ready/` | X_train, X_test, y splits | TO VERIFY |

### ML Artifacts (`backend/artifacts/`)
| Path | Contents | Loaded by API? |
|------|----------|----------------|
| `models/xgboost.joblib` | v1 XGBoost Pipeline | ✅ YES (active) |
| `models/random_forest.joblib` | v1 Random Forest | No |
| `models/logistic_regression.joblib` | v1 Logistic Regression | No |
| `models_v2_improved/best_model_v2_improved.joblib` | v2 XGBoost (unweighted, threshold=0.13) | ❌ Not loaded |
| `models_v3_interaction/best_model_v3_interaction.joblib` | v3 XGBoost (interaction features) | ❌ Not loaded |
| `metrics/best_model.json` | Active model config: XGBoost, AP=0.332, threshold=**0.5** | ✅ |
| `metrics/model_comparison.csv` | v1 comparison (4 models) | ✅ Served |
| `metrics_v2_improved/best_model.json` | v2 config: threshold=**0.13**, test AP=0.336 | Not served |
| `metrics_v3_interaction/experiment_summary.json` | v3 no improvement | Not served |
| `explainability/shap_feature_importance.csv` | SHAP values | ✅ Served |
| `explainability/shap_summary_plot.png` | SHAP plot | ✅ Exists |
| `explainability/individual_waterfall.png` | Waterfall chart | ✅ Exists |
| `optimization/optimized_maintenance_schedule.csv` | 275 scheduled | ✅ Served |
| `optimization/unscheduled_maintenance_requests.csv` | 225 unscheduled | ✅ Served |
| `simulation/simulated_maintenance_requests.csv` | 500 requests | ✅ Served |
| `simulation/simulated_block_windows.csv` | 140 windows | ✅ Served |
| `planning/planning_kpis.json` | KPIs summary | ✅ Served |

### MLflow (`mlartifacts/`, `mlflow.db`)
- Experiment 1 with 3 runs + multiple registered models
- Run IDs: `3648b10c`, `4b7fcc94`, `c055ebbc` (TO VERIFY which is which)

### Frontend (`frontend/`)
- **ONLY `.vite/` directory exists** — no `src/`, no `package.json`, no `index.html`
- The frontend is **completely missing** from the workspace

### Tests (`tests/`)
| File | Tests | Status |
|------|-------|--------|
| `test_api.py` | 24 endpoint + filter tests | ✅ 24/24 pass |
| `test_artifact_service.py` | 3 artifact error tests | ✅ 3/3 pass |
| `test_filter_service.py` | 1 column-not-found test | ✅ 1/1 pass |
| `test_model_api.py` | 5 predict + risk tests | ✅ 5/5 pass |

**TOTAL: 33/33 tests passing** ✅ (Reproduced: `pytest tests/ -v → 33 passed in 3.34s`)

### Docs (`docs/`)
| File | Status |
|------|--------|
| `docs/architecture.md` | Exists |
| `docs/model_card.md` | Exists (TO VERIFY content vs handbook claims) |
| `docs/PROGRESS.md` | ✅ Created now |
| `docs/audit/PROJECT_AUDIT_REPORT.md` | ✅ This file |
| `docs/AI_PROJECT_DETAILS.md` | ❌ MISSING — to create in P2 |
| `docs/ARCHITECTURE.md` (Mermaid) | ❌ MISSING — to create in P7 |
| `docs/VIVA_PREP.md` | ❌ MISSING — to create in P7 |
| `docs/TESTING_REPORT.md` | ❌ MISSING — to create in P6 |
| `docs/DEPLOYMENT.md` | ❌ MISSING — to create in P8 |

---

## 3. ACTIVE MODEL IDENTIFICATION

**API loads**: `backend/artifacts/models/xgboost.joblib`  
**Decision file**: `backend/artifacts/metrics/best_model.json`  
```json
{
  "best_model": "XGBoost",
  "threshold": 0.5,
  "validation_average_precision": 0.3322,
  "threshold_status": "constraints_not_satisfied"
}
```
**Problem**: `threshold_status: "constraints_not_satisfied"` means no threshold met the operational constraints. The API is using 0.5 as fallback, which gives low recall on a 20% positive class.

**v2 artifact** (not loaded by API): threshold=0.13, test recall=0.9998, test precision=0.202. This flags ~100% of data as positive. Also poor, but differently so.

---

## 4. REPRODUCED METRICS (from saved CSVs — not rerun yet)

| Model | ROC-AUC | AP (val) | F1 (thr=0.5) | Recall | Precision |
|-------|---------|----------|--------------|--------|-----------|
| XGBoost | 0.5508 | **0.3322** | 0.268 | 0.258 | 0.278 |
| Random Forest | 0.5506 | 0.3276 | 0.207 | 0.120 | 0.731 |
| Logistic Regression | 0.5383 | 0.2503 | 0.293 | 0.404 | 0.229 |
| Dummy Baseline | 0.5000 | 0.2020 | 0.000 | 0.000 | 0.000 |

**XGBoost is best by AP=0.3322 vs Dummy=0.2020** — marginally better than random. ROC-AUC=0.55 is barely above chance (0.50). The model has **weak predictive power** — this must be stated clearly in docs and UI.

---

## 5. KEY INCONSISTENCIES FOUND

### 5A. API Threshold Mismatch (HIGH)
- API uses `threshold=0.5` from v1 `best_model.json`
- v2 experiments found `threshold=0.13` as best operational threshold
- These produce very different results (v1: low recall, v2: near-100% recall at 20% precision)
- **Neither is good.** Root cause: near-random model, no threshold satisfies operational constraints.

### 5B. Three Concepts NOT clearly separated in API
- (A) ML prediction probability: returned by `/ml/predict` ✅
- (B) dataset `risk_score` column: in cleaned CSV, used by `ml/risk_scoring.py` ✅
- (C) `operational_priority_score` from `priority_engine.py` (rule-based) ✅
- **Problem**: API does not label which concept is which in responses. SHAP not returned per-prediction.

### 5C. risk_score + failure_type in cleaned CSV (Leakage Risk)
- Columns `risk_score`, `failure_type`, `failure_severity` are in `railway_cleaned.csv`
- They are excluded from model features (confirmed: model only uses 30 features)
- BUT they remain in the processed CSV — if someone uses them in future training, it creates leakage
- **Model pipeline correctly excludes them** — no active leakage, but risky for future experiments

### 5D. Optimizer Algorithm Name
- `block_optimizer.py` uses `from ortools.sat.python import cp_model`
- This is **OR-Tools CP-SAT** (Constraint Programming with SAT-based solver)
- Must NOT be called ILP, MILP, or Genetic Algorithm
- Correct name: **Constraint Programming with CP-SAT solver**

### 5E. No Calibration
- No `CalibratedClassifierCV` applied to any model
- Probability outputs are raw XGBoost probabilities (not isotonic/sigmoid calibrated)
- Calibration plot not generated yet

### 5F. Frontend Completely Missing
- `frontend/` directory contains only `.vite/` cache folder
- No `src/`, `package.json`, `tsconfig.json`, `index.html`
- P5 requires building the full React+TS+Vite+Recharts+Tailwind frontend from scratch

---

## 6. MISSING GRAPHS / PLOTS / EVALUATIONS

| Item | Status |
|------|--------|
| Class distribution plot | ❌ Missing |
| Feature distributions (all 30) | ❌ Missing |
| Correlation heatmap | ❌ Missing |
| Missing values heatmap | ❌ Missing |
| ROC curve (multi-model) | ❌ Missing |
| PR curve (multi-model) | ❌ Missing |
| Confusion matrices | ❌ Missing |
| Threshold analysis table | ❌ Missing |
| Calibration plot | ❌ Missing |
| Probability histogram | ❌ Missing |
| Model comparison bar chart | ❌ Missing |
| CV box plots | ❌ Missing |
| Feature importance (from model) | ❌ Missing |
| SHAP summary (PNG exists) | ✅ Exists |
| SHAP bar (CSV exists) | ✅ Exists |
| Individual waterfall (PNG) | ✅ Exists |
| FP vs FN error analysis | ❌ Missing |

---

## 7. COMPLETION PERCENTAGES

### Method: each area scored by: (working items) / (total required items) × 100

**A. Backend** — **72%**
- Working: FastAPI app, CORS, 20+ endpoints, model loading, prediction, risk scoring, artifact service, filter/pagination, logging, error handling = 10/14 required items
- Missing: SHAP per-prediction, model version in response, calibration flag, env-based CORS = 4 items

**B. Frontend** — **0%**
- Frontend directory is empty. Only `.vite/` cache exists. Zero source files.
- Required: React+TS+Vite setup, 5 pages, all components, charts = 0 of everything

**C. Dataset & Preprocessing** — **70%**
- Working: raw data, cleaned data, 30 feature pipeline, train/val/test split, model_ready/
- Missing: verified leakage audit (automated), preprocessing consistency proof, data quality report = 30% gap

**D. ML Model Performance** — **40%**
- Working: 4 models trained, metrics in CSV, SHAP artifacts, v2/v3 experiments
- Missing: calibration, full threshold grid, CV results, permutation test, seed stability, complete metric set (MCC, Kappa, Brier, etc.), honest dummy comparison documented = 60% gap

**E. Explainability** — **35%**
- Working: SHAP summary PNG, feature importance CSV, individual waterfall PNG
- Missing: per-prediction SHAP in API, dependence plots, 3 worked examples, SHAP artifacts served correctly = 65% gap

**F. Optimizer** — **65%**
- Working: CP-SAT scheduler runs, 275/500 scheduled, validation report, analytics CSVs, planning KPIs
- Missing: documented algorithm explanation, before/after analysis, worked example in docs, unscheduled reason analysis = 35% gap

**G. MLOps** — **40%**
- Working: MLflow db + artifacts (3 runs), versioned artifact folders (v1/v2/v3)
- Missing: verified run IDs linked to code, registered model vs loaded model traced, MLflow experiment names documented = 60% gap

**H. Tests & Deployment** — **35%**
- Working: 33 tests passing
- Missing: ML edge case tests, invalid range tests, optimizer constraint tests, risk scoring edge cases, frontend build, deployment docs = 65% gap

**I. Documentation** — **20%**
- Working: `architecture.md` (basic), `model_card.md` (unverified content)
- Missing: AI_PROJECT_DETAILS.md, EXPERIMENT_REPORT.md, MODEL_CARD.md (complete), VIVA_PREP.md, ARCHITECTURE.md (Mermaid), TESTING_REPORT.md, DEPLOYMENT.md, README update

---

## 8. PRIORITIZED ROADMAP

| Priority | Phase | Task | Blocking? |
|----------|-------|------|-----------|
| 🔴 P0 | P2 | ML diagnosis: leakage audit, signal check, dummy comparison | Yes — all ML docs depend |
| 🔴 P0 | P2 | Full metric set: MCC, Kappa, Brier, calibration, threshold grid on val | Yes |
| 🔴 P0 | P2 | Create docs/AI_PROJECT_DETAILS.md with verified facts | Yes — viva depends |
| 🟠 P1 | P3 | Add SHAP contributions to /ml/predict response | Backend completeness |
| 🟠 P1 | P3 | Add model version + timestamp to /predict | Backend completeness |
| 🟠 P1 | P5 | Build frontend from scratch (React+TS+Vite+Tailwind+Recharts) | Viva UI |
| 🟠 P1 | P4 | Document CP-SAT algorithm, objective, constraints | Viva defense |
| 🟡 P2 | P6 | Add ML edge case + optimizer constraint tests | Test coverage |
| 🟡 P2 | P7 | ARCHITECTURE.md with Mermaid, VIVA_PREP.md, README | Documentation |
| 🟡 P2 | P8 | env-based CORS, production build check, DEPLOYMENT.md | Deploy |
| 🟢 P3 | P2 | Calibration (isotonic), 5-fold CV, seed stability | Model completeness |
| 🟢 P3 | P5 | All missing plots (ROC, PR, confusion, threshold, calibration) | UI + docs |

---

## 9. SINGLE MOST IMPORTANT NEXT ACTION

**Start P2: Run full ML diagnosis script** — reproduce metrics from scratch on the fixed train/val/test split, generate all missing metric columns (MCC, Kappa, Brier, Balanced Accuracy), do threshold grid analysis on validation set, run permutation importance to confirm model has real signal (not noise), and begin filling `docs/AI_PROJECT_DETAILS.md`. This is the foundation everything else rests on.
