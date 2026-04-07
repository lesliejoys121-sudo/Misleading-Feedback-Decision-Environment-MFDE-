from pydantic import BaseModel, Field
from typing import Optional


class Observation(BaseModel):
    observed_value: float = Field(..., description="Telemetry output from degraded reactor sensors, distorted by noise and potential electrical interference.")
    noise_level: float = Field(..., description="Current standard deviation of known sensor noise (0.0 to 1.0)")
    step: int = Field(default=0, description="Current step number")


class Action(BaseModel):
    prediction: float = Field(..., description="Agent's estimation of the true reactor core temperature (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Agent's confidence in its temperature estimation (0–1)")


class Reward(BaseModel):
    value: float = Field(..., description="Final reward value, clamped to [0, 1]")
    accuracy_reward: float = Field(..., description="Reward component for prediction accuracy")
    calibration_penalty: float = Field(..., description="Penalty for miscalibrated confidence")
    actual_accuracy: float = Field(..., description="True accuracy of the prediction")
