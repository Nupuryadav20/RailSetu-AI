# RailSetu AI - Database Schema Documentation

This document describes the relational database layer for RailSetu AI using SQLAlchemy 2.x and SQLite (`data/railsetu.db`).

## Overview

The database stores operational simulation data, optimizer schedules, model experiment telemetry, performance metrics across splits, threshold analysis, feature importance rankings, and real-time prediction history.

## Mermaid Entity-Relationship Diagram

```mermaid
erDiagram
    MAINTENANCE_REQUESTS {
        int id PK
        string request_id UK
        string corridor_id
        string department
        string maintenance_type
        float required_duration_hours
        int team_size_required
        string earliest_start
        string deadline
        float priority_score
        string priority_level
    }

    BLOCK_WINDOWS {
        int id PK
        string window_id UK
        string corridor_id
        string window_start
        string window_end
        string window_status
        int max_parallel_teams
    }

    SCHEDULE_ASSIGNMENTS {
        int id PK
        string request_id FK
        string window_id FK
        string corridor_id
        string department
        string scheduled_start
        string scheduled_end
        float priority_score
    }

    UNSCHEDULED_REQUESTS {
        int id PK
        string request_id FK
        string corridor_id
        string department
        float priority_score
        string priority_level
        string reason
    }

    PREDICTIONS {
        int id PK
        string timestamp
        string model_name
        string model_version
        string input_json
        float probability
        float threshold
        int predicted_class
        string top_shap_json
        string warnings_json
    }

    MODEL_RUNS {
        int id PK
        string mlflow_run_id
        string model_name
        string version
        string params_json
        string created_at
        boolean is_served
    }

    MODEL_METRICS {
        int id PK
        string model_name
        string version
        string split
        string metric
        float value
    }

    THRESHOLD_METRICS {
        int id PK
        string version
        float threshold
        float precision
        float recall
        float f1
        float f2
        int tp
        int fp
        int tn
        int fn
        float pct_flagged
        string operational_character
    }

    FEATURE_IMPORTANCE {
        int id PK
        string version
        string feature
        float importance_perm
        float importance_tree
        float mean_abs_shap
    }

    FEATURE_STATS {
        int id PK
        string feature UK
        string data_type
        float min_val
        float max_val
        float mean_val
        float std_val
        float p25
        float p50
        float p75
        float class_0_mean
        float class_1_mean
        string categories_json
    }

    DATASET_INFO {
        int id PK
        string key UK
        string value_json
    }

    MAINTENANCE_REQUESTS ||--o| SCHEDULE_ASSIGNMENTS : "assigned to"
    BLOCK_WINDOWS ||--o{ SCHEDULE_ASSIGNMENTS : "contains"
    MAINTENANCE_REQUESTS ||--o| UNSCHEDULED_REQUESTS : "unassigned"
```

## Seeding & Idempotency

The database is seeded via `python -m scripts.seed_db`. The seeder:
1. Re-initializes all tables if missing (`init_db()`).
2. Clears existing data to ensure full idempotency.
3. Ingests simulated requests (500) and block windows (140).
4. Populates CP-SAT optimization results (275 scheduled, 225 unscheduled).
5. Populates model runs, all train/val/test evaluation metrics for v4 and v5 models, threshold evaluations, and feature rankings.
6. Computes distribution percentiles, min/max/std, and class conditional means for all 30 features from training data for runtime z-score and anomaly detection.
