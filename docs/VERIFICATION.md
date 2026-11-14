# RailSetu AI — System Verification & Quality Audit Report

**Audit Date:** 2026-11-14 20:14:43  
**Target Environment:** Local Python (.venv) + FastAPI (`http://127.0.0.1:8000`) + SQLite (`data/railsetu.db`)  
**Served Model:** XGBoost Pipeline (`v5`) with Calibration & SHAP Diagnostics

---

## 1. Automated Endpoint Test Matrix

All 18 key API endpoints were tested against the live backend server:

| Endpoint | Method | Expected | Actual | Content-Type | Status |
|---|---|---|---|---|---|
| `/health` | `GET` | 200 | 200 | application/json | PASS |
| `/api/dashboard/summary` | `GET` | 200 | 200 | application/json | PASS |
| `/api/maintenance/requests?page=1&page_size=5` | `GET` | 200 | 200 | application/json | PASS |
| `/api/maintenance/requests?department=Engineering&priority=High` | `GET` | 200 | 200 | application/json | PASS |
| `/api/schedule/optimized` | `GET` | 200 | 200 | application/json | PASS |
| `/api/schedule/unscheduled` | `GET` | 200 | 200 | application/json | PASS |
| `/api/planning/weekly` | `GET` | 200 | 200 | application/json | PASS |
| `/api/planning/monthly` | `GET` | 200 | 200 | application/json | PASS |
| `/api/simulation/block-windows` | `GET` | 200 | 200 | application/json | PASS |
| `/api/ml/v5/comparison?split=test` | `GET` | 200 | 200 | application/json | PASS |
| `/api/ml/v5/thresholds` | `GET` | 200 | 200 | application/json | PASS |
| `/api/ml/v5/cv` | `GET` | 200 | 200 | application/json | PASS |
| `/api/ml/v5/feature-selection` | `GET` | 200 | 200 | application/json | PASS |
| `/api/ml/v5/feature-stats` | `GET` | 200 | 200 | application/json | PASS |
| `/api/ml/v5/experiments` | `GET` | 200 | 200 | application/json | PASS |
| `/api/ml/v5/metadata` | `GET` | 200 | 200 | application/json | PASS |
| `/api/ml/predictions?limit=5` | `GET` | 200 | 200 | application/json | PASS |
| `/api/ml/predictions/download?limit=10` | `GET` | 200 | 200 | text/csv | PASS |

**Overall Endpoint Pass Rate:** 18/18 (100.0%)

---

## 2. ML Inference & Diagnostics Verification (`/api/ml/predict`)

### 2.1 Nominal Prediction Payload
Input: 30 standardized railway telemetry features within historical distributions.

- **Model Identity:** `XGBoost (v5)`
- **Raw Probability:** `45.75%`
- **Calibrated Probability:** `18.48%`
- **Operating Threshold:** `0.45 (balanced_capacity_screening_0.45)`
- **Predicted Class:** `1` (`MAINTENANCE REQUIRED`)
- **Top SHAP Contributors:** `['track_curvature_degree (-0.064)', 'train_type (+0.049)', 'track_vibration_level (-0.047)', 'rail_wear_mm (-0.042)']`
- **Data Quality Warnings:** `[]`

### 2.2 Out-of-Distribution (OOD) Input Test
Input: Anomalous high temperatures (`axle_temperature_c=180.0°C`, `ambient_temperature_c=85.0°C`, `rail_wear_mm=18.0mm`).

- **Raw Probability:** `37.99%`
- **Calibrated Probability:** `18.27%`
- **Operating Threshold:** `0.45`
- **Predicted Class:** `0` (`NO MAINTENANCE REQUIRED`)
- **Triggered Warnings:**
  - `ambient_temperature_c (85.0) exceeds training max (50.0)`
  - `axle_temperature_c (180.0) exceeds training max (117.66216220204878)`
  - `rail_wear_mm has unusual z-score (3.35)`
- **High Z-Score Anomalies Flagged (|z| > 3):** `['ambient_temperature_c (z=6.95)', 'axle_temperature_c (z=9.55)', 'rail_wear_mm (z=3.35)']`

### 2.3 Invalid Schema Handling
- **Missing Fields HTTP Status:** `422` (Properly caught by schema validation)

---

## 3. Database Persistence & Prediction History

- **Total Logged Predictions in SQLite (`data/railsetu.db`):** `44` records
- **Recent Log Entries:**
  - Record ID `44`: Model `v5`, Prob `0.3799`, Label `0`, Threshold `0.45`, Timestamp `2026-11-14T14:44:43.425726+00:00`
  - Record ID `43`: Model `v5`, Prob `0.4575`, Label `1`, Threshold `0.45`, Timestamp `2026-11-14T14:44:43.398438+00:00`
  - Record ID `42`: Model `v5`, Prob `0.4575`, Label `1`, Threshold `0.45`, Timestamp `2026-11-14T14:44:23.545540+00:00`

---

## 4. Cross-System Artifact Consistency

| Check | Metadata Source | Target Documentation / System | Result |
|---|---|---|---|
| Best Model Identity | `XGBoost` | `docs/MODEL_CARD.md` | MATCH |
| Serving Model Version | `v5` | FastAPI `/ml/predict` | MATCH |
| Validation PR-AUC | `0.3320` | Model Card Table | MATCH |
| Test PR-AUC | `0.3296` | Model Card Table | MATCH |
| Chosen Threshold | `0.45` | UI Default & API | MATCH |

---

## 5. Architectural Concept Separation Audit

1. **Concept A (ML Failure Probability):**
   - Output of statistical model (`0.0` - `1.0`).
   - Calibrated via validation sigmoid curve.
   - Distinctly labelled with uncertainty disclaimer.
2. **Concept B (Dataset `risk_score`):**
   - Pre-existing synthetic baseline attribute in raw dataset.
   - Strictly sequestered from model training features to prevent data leakage.
3. **Concept C (Operational Priority Score):**
   - Deterministic rule-based index combining department urgency, corridor traffic density, and track safety flags.
   - Independent of ML probability.

---

## 6. Frontend Build & Static Analysis
- **TypeScript & Vite Compilation:** Zero compile errors (`tsc -b && vite build` built in ~130ms).
- **CSS Audit:** Zero missing or unstyled BEM classes.
- **Responsiveness:** Validated on desktop (1440x950) and mobile (390x844).
