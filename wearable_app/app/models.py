from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class UserData(BaseModel):
    gender: str
    age: int
    occupation: str
    sleep_duration: float
    physical_activity: float
    bmi: float
    daily_steps: int
    stress_level: int
    blood_pressure: int

class PredictionData(BaseModel):
    timestamp: datetime
    sleep_disorder: str
    sleep_quality: str
    sleep_duration: float

class UserPredictionData(UserData):
    pass

class GyroscopeData(BaseModel):
    distance: float
    timestamp: Optional[datetime] = None
