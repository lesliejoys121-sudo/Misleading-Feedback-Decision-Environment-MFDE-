import random
from typing import Any, Dict, Optional, Tuple

from models import Action, Observation, Reward
from tasks import Task, TASK_MAP


class MFDEEnvironment:
    """
    Misleading Feedback Decision Environment (MFDE) -> Industrial Sensor Calibration.

    The agent acts as a control system predicting true reactor core temperature 
    based on noisy/biased telemetry from degraded sensors, and must estimate its confidence.
    """

    def __init__(self, task_name: str = "easy"):
        if task_name not in TASK_MAP:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASK_MAP.keys())}")
        self.task: Task = TASK_MAP[task_name]
        self._rng = random.Random(self.task.seed)
        self._step_count = 0
        self._hidden_truth = 0.0
        self._observed_value = 0.0
        self._done = False

        # History for grading
        self._predictions: list = []
        self._hidden_truths: list = []
        self._confidences: list = []

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset environment to initial state and return first observation."""
        self._rng = random.Random(self.task.seed)  # deterministic reset
        self._step_count = 0
        self._done = False
        self._predictions = []
        self._hidden_truths = []
        self._confidences = []
        self._hidden_truth, self._observed_value = self._generate_values()
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process one action and return (observation, reward, done, info).
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        # Record history
        self._predictions.append(action.prediction)
        self._hidden_truths.append(self._hidden_truth)
        self._confidences.append(action.confidence)

        # Compute reward
        reward = self._compute_reward(action)

        self._step_count += 1

        # Advance to next state
        if self._step_count >= self.task.max_steps:
            self._done = True
            obs = self._make_observation()
        else:
            self._hidden_truth, self._observed_value = self._generate_values()
            obs = self._make_observation()

        info = {
            "hidden_truth": self._hidden_truth,
            "step": self._step_count,
            "task": self.task.name,
        }

        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return the internal (hidden) state of the environment."""
        return {
            "hidden_truth": self._hidden_truth,
            "observed_value": self._observed_value,
            "noise_level": self.task.noise_std,
            "bias": self.task.bias,
            "step": self._step_count,
            "done": self._done,
            "task": self.task.name,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_values(self) -> Tuple[float, float]:
        """Generate a new hidden truth and its noisy observation."""
        if self._step_count == 0:
            hidden = self._rng.uniform(0.0, 1.0)
        else:
            drift = self._rng.uniform(-self.task.drift_strength, self.task.drift_strength)
            hidden = max(0.0, min(1.0, self._hidden_truth + drift))
            
        noise = self._rng.gauss(0, self.task.noise_std)
        observed = hidden + noise + self.task.bias
        
        # Occasionally introduce deceptive observations
        if self.task.deception_probability > 0 and self._rng.random() < self.task.deception_probability:
            observed += self._rng.uniform(-0.3, 0.3)

        # Clamp observed to [-1, 2] range to avoid extreme outliers
        observed = max(-1.0, min(2.0, observed))
        return round(hidden, 4), round(observed, 4)

    def _make_observation(self) -> Observation:
        return Observation(
            observed_value=self._observed_value,
            noise_level=self.task.noise_std,
            step=self._step_count,
        )

    def _compute_reward(self, action: Action) -> Reward:
        accuracy = max(0.0, 1.0 - abs(action.prediction - self._hidden_truth))
        calibration = max(0.0, 1.0 - abs(action.confidence - accuracy))
        raw_reward = 0.6 * accuracy + 0.4 * calibration
        final_reward = float(max(0.0, min(1.0, raw_reward)))

        return Reward(
            value=round(final_reward, 4),
            accuracy_reward=round(accuracy, 4),
            calibration_penalty=round(calibration, 4), # kept field name for model compatibility
            actual_accuracy=round(accuracy, 4),
        )

    def get_history(self) -> Dict[str, list]:
        return {
            "predictions": self._predictions,
            "hidden_truths": self._hidden_truths,
            "confidences": self._confidences,
        }
