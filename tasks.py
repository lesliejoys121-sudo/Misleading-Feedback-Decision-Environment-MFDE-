from dataclasses import dataclass
from typing import List


@dataclass
class Task:
    name: str
    description: str
    noise_std: float        # Standard deviation of random noise
    bias: float             # Systematic bias added to observations
    drift_strength: float   # Max drift per step
    deception_probability: float # Probability of misleading spike
    max_steps: int
    seed: int


TASKS: List[Task] = [
    Task(
        name="easy",
        description="Low noise, mostly accurate observations. Learn basic prediction.",
        noise_std=0.05,
        bias=0.0,
        drift_strength=0.0,
        deception_probability=0.0,
        max_steps=10,
        seed=42,
    ),
    Task(
        name="medium",
        description="Moderate noise, occasional misleading signals. Balance trust and correction.",
        noise_std=0.20,
        bias=0.10,
        drift_strength=0.05,
        deception_probability=0.20,
        max_steps=10,
        seed=123,
    ),
    Task(
        name="hard",
        description="High noise + bias, observations frequently misleading. Learn to distrust environment.",
        noise_std=0.40,
        bias=0.30,
        drift_strength=0.10,
        deception_probability=0.40,
        max_steps=10,
        seed=999,
    ),
]

TASK_MAP = {t.name: t for t in TASKS}
