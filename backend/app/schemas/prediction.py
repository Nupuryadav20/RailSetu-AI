
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    region: str
    season: str
    train_type: str

    train_age_years: float = Field(ge=0)
    average_speed_kmph: float = Field(ge=0)
    distance_travelled_km: float = Field(ge=0)
    track_temperature_c: float
    rail_wear_mm: float = Field(ge=0)
    track_vibration_level: float = Field(ge=0)
    ballast_condition: str
    track_curvature_degree: float
    ambient_temperature_c: float
    humidity_percent: float = Field(ge=0, le=100)
    rainfall_mm: float = Field(ge=0)
    wind_speed_kmph: float = Field(ge=0)
    wheel_wear_percent: float = Field(ge=0, le=100)
    axle_temperature_c: float
    brake_pressure_psi: float = Field(ge=0)
    brake_pad_wear_percent: float = Field(ge=0, le=100)
    bearing_temperature_c: float
    battery_voltage: float = Field(ge=0)
    traction_motor_temp_c: float
    signal_system_status: str
    power_consumption_kw: float = Field(ge=0)
    load_factor_percent: float = Field(ge=0, le=100)
    daily_trips: float = Field(ge=0)
    delay_minutes: float = Field(ge=0)
    last_maintenance_days: float = Field(ge=0)
    inspection_score: float = Field(ge=0, le=100)
    sensor_health_index: float = Field(ge=0, le=100)


class PredictionRequest(BaseModel):
    records: list[PredictionRecord] = Field(min_length=1, max_length=100)


class RiskScoringRequest(BaseModel):
    records: list[PredictionRecord] = Field(min_length=1, max_length=100)
