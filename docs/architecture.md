# RailSetu AI — Architecture Specification
_System Design & Component Interaction Guide_

---

## 1. High-Level System Architecture

RailSetu AI integrates predictive maintenance machine learning, constraint-based maintenance block window scheduling, and real-time operational decision support.

```mermaid
flowchart TD
    subgraph Data Layer
        A1[Kaggle Synthetic Dataset 100K] --> A2[Data Cleaning & Deduplication]
        A2 --> A3[railway_cleaned.csv 100k x 35]
        A3 --> A4[Leakage Column Exclusion]
        A4 --> A5[model_ready Train/Val/Test Splits 60k/20k/20k]
    end

    subgraph ML Pipeline
        A5 --> M1[Preprocessing: Imputer + Scaler + OrdinalEncoder]
        M1 --> M2[Model Training: XGBoost / RF / GBDT / LogReg]
        M2 --> M3[Threshold Tuning on Validation Set]
        M3 --> M4[Model Artifacts & Metrics v4]
        M4 --> M5[SHAP Feature Explanations]
    end

    subgraph Optimization Engine
        O1[Simulated Maintenance Requests 500] --> O3[Operational Priority Engine]
        O2[Simulated Block Windows 140] --> O4[OR-Tools CP-SAT Solver]
        O3 --> O4
        O4 --> O5[Optimized Maintenance Schedule 275]
        O4 --> O6[Unscheduled Requests 225]
    end

    subgraph Backend Service Layer - FastAPI
        M4 --> B1[Model Service /ml/predict]
        M5 --> B1
        O3 --> B2[Risk Service /risk/score]
        O5 --> B3[Schedule APIs /api/schedule/*]
        M4 --> B4[Metrics & Plot Service /api/ml/*]
    end

    subgraph Frontend Application - React + Vite + Tailwind
        B1 --> F1[Predictive Maintenance View]
        B2 --> F2[Operational Risk Dashboard]
        B3 --> F3[Block Window Scheduler Gantt]
        B4 --> F4[Model Insights & Diagnostics]
    end
```

---

## 2. Distinction of Three Core Concepts

```mermaid
flowchart LR
    subgraph Concept A - Statistical
        CA1[30 Telemetry Features] --> CA2[XGBoost Pipeline]
        CA2 --> CA3[Maintenance Probability 0-100%]
    end

    subgraph Concept B - Dataset Label
        CB1[Raw Synthetic Data] --> CB2[Column: risk_score]
        CB2 --> CB3[Excluded from ML Training to Avoid Leakage]
    end

    subgraph Concept C - Operational Decision
        CA3 --> CC1[Operational Priority Engine]
        CC2[Equipment Condition Score] --> CC1
        CC3[Corridor Urgency & Delays] --> CC1
        CC1 --> CC4[Priority Score 0-100 & High/Med/Low]
        CC4 --> CC5[Input Weight for CP-SAT Optimizer]
    end
```

---

## 3. CP-SAT Maintenance Block Optimizer Workflow

```mermaid
sequenceDiagram
    autonumber
    participant Requester as Maintenance Departments
    participant PriorityEngine as Priority Engine (Concept C)
    participant CPSATSolver as OR-Tools CP-SAT Solver
    participant ScheduleStore as Schedule Artifacts
    participant Dispatcher as Section Controller UI

    Requester->>PriorityEngine: Submit Maintenance Requests (Duration, Corridor, Department)
    PriorityEngine->>PriorityEngine: Compute Priority Score (Model Prob + Condition + Safety)
    PriorityEngine->>CPSATSolver: Formulate Interval Variables & Constraints
    CPSATSolver->>CPSATSolver: Apply No-Overlap, Crew Limits, Window Boundaries
    CPSATSolver->>CPSATSolver: Maximize Total Scheduled Priority
    CPSATSolver->>ScheduleStore: Persist 275 Scheduled & 225 Unscheduled Requests
    ScheduleStore->>Dispatcher: Render Scheduled Corridor Gantt & Conflict Report
```

---

## 4. API Service Layout & Routing

- `/api/dashboard/summary`: High-level operational KPIs, scheduled vs unscheduled counts.
- `/api/maintenance/requests`: Paginated and filterable maintenance request backlog.
- `/api/schedule/optimized`: Optimized block window assignments.
- `/api/schedule/unscheduled`: Unallocated jobs with rejection reasons (capacity, duration overflow).
- `/api/ml/predict`: Single or batch inference returning failure likelihood ($0-100\%$), decision class, and top-5 local SHAP contributors.
- `/api/risk/score`: Blended operational priority scoring combining model output with mechanical wear indicators.
- `/api/ml/metrics/*`: Transparent access to master metrics, validation/test matrices, and threshold trade-off curves.
- `/api/ml/plots/{name}`: Direct delivery of diagnostic figures (ROC, PR, Calibration, SHAP).
- `/api/ml/v5/*`: V5 active model metrics, feature selection ablation curves, repeated CV summaries, feature distribution stats, and MLflow run history.
- `/api/ml/predictions`: Query stored predictions from SQLite database with class and probability filtering.
- `/api/ml/predictions/download`: CSV export download of saved predictions.

---

## 5. Relational Database Layer (SQLAlchemy 2 + SQLite)

```mermaid
flowchart TD
    subgraph Storage Layer
        DB[(data/railsetu.db SQLite)]
    end

    subgraph Database Models
        M_REQ[maintenance_requests]
        M_WIN[block_windows]
        M_SCH[schedule_assignments]
        M_UNS[unscheduled_requests]
        M_PRED[predictions]
        M_RUN[model_runs]
        M_MET[model_metrics]
        M_THR[threshold_metrics]
        M_STAT[feature_stats]
    end

    subgraph Service Interactions
        SEED[scripts/seed_db.py] -->|Seed Simulation & Metrics| DB
        PREDICT[/api/ml/predict] -->|Persist Inferences| M_PRED
        PREDICT -->|Read Training Stats for Z-Score| M_STAT
        HISTORY[/api/ml/predictions] -->|Query History| M_PRED
        GOV[/api/ml/v5/experiments] -->|Query MLflow Runs| M_RUN
    end

    DB --- M_REQ
    DB --- M_WIN
    DB --- M_SCH
    DB --- M_UNS
    DB --- M_PRED
    DB --- M_RUN
    DB --- M_MET
    DB --- M_THR
    DB --- M_STAT
```

