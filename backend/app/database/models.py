from sqlalchemy import Column, Integer, Float, String, Text, Boolean, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class MaintenanceRequest(Base):
    __tablename__ = "maintenance_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), index=True, nullable=False)
    corridor_id = Column(String(64), index=True)
    department = Column(String(64), index=True)
    maintenance_type = Column(String(128))
    required_duration_hours = Column(Float)
    team_size_required = Column(Integer)
    earliest_start = Column(String(64))
    deadline = Column(String(64))
    priority_score = Column(Float)
    priority_level = Column(String(32), index=True)


class BlockWindow(Base):
    __tablename__ = "block_windows"

    id = Column(Integer, primary_key=True, autoincrement=True)
    window_id = Column(String(64), index=True, nullable=False)
    corridor_id = Column(String(64), index=True)
    window_start = Column(String(64))
    window_end = Column(String(64))
    window_status = Column(String(32))
    max_parallel_teams = Column(Integer)


class ScheduleAssignment(Base):
    __tablename__ = "schedule_assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), index=True, nullable=False)
    window_id = Column(String(64), index=True)
    corridor_id = Column(String(64))
    department = Column(String(64))
    scheduled_start = Column(String(64))
    scheduled_end = Column(String(64))
    priority_score = Column(Float)


class UnscheduledRequest(Base):
    __tablename__ = "unscheduled_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), index=True, nullable=False)
    corridor_id = Column(String(64))
    department = Column(String(64))
    priority_score = Column(Float)
    priority_level = Column(String(32))
    reason = Column(Text)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String(64), index=True, nullable=False)
    model_name = Column(String(64))
    model_version = Column(String(32), index=True)
    input_json = Column(Text)
    probability = Column(Float)
    threshold = Column(Float)
    predicted_class = Column(Integer, index=True)
    top_shap_json = Column(Text)
    warnings_json = Column(Text)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mlflow_run_id = Column(String(64), index=True)
    model_name = Column(String(64), index=True)
    version = Column(String(32), index=True)
    params_json = Column(Text)
    created_at = Column(String(64))
    is_served = Column(Boolean, default=False)


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(64), index=True)
    version = Column(String(32), index=True)
    split = Column(String(32), index=True)
    metric = Column(String(64), index=True)
    value = Column(Float)


class ThresholdMetric(Base):
    __tablename__ = "threshold_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(32), index=True)
    threshold = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1 = Column(Float)
    f2 = Column(Float)
    tp = Column(Integer)
    fp = Column(Integer)
    tn = Column(Integer)
    fn = Column(Integer)
    pct_flagged = Column(Float)
    operational_character = Column(String(256))


class FeatureImportance(Base):
    __tablename__ = "feature_importance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(32), index=True)
    feature = Column(String(64), index=True)
    importance_perm = Column(Float)
    importance_tree = Column(Float)
    mean_abs_shap = Column(Float)


class FeatureStat(Base):
    __tablename__ = "feature_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feature = Column(String(64), index=True, nullable=False)
    data_type = Column(String(32))
    min_val = Column(Float)
    max_val = Column(Float)
    mean_val = Column(Float)
    std_val = Column(Float)
    p25 = Column(Float)
    p50 = Column(Float)
    p75 = Column(Float)
    class_0_mean = Column(Float)
    class_1_mean = Column(Float)
    categories_json = Column(Text)


class DatasetInfo(Base):
    __tablename__ = "dataset_info"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(64), unique=True, index=True, nullable=False)
    value_json = Column(Text)
