import random
from typing import Any, Dict, Optional, Tuple

from models import Action, Observation, Reward
from tasks import Task, TASK_MAP


class MFDEEnvironment:
    """
    Misleading Feedback Decision Environment (MFDE).

    The agent sees a noisy/biased version of the hidden truth,
    must predict the true value, and estimate its confidence.
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
        hidden = self._rng.uniform(0.0, 1.0)
        noise = self._rng.gauss(0, self.task.noise_std)
        observed = hidden + noise + self.task.bias
        # Clamp observed to [0, 1] range to avoid extreme outliers
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
        calibration_penalty = -abs(action.confidence - accuracy)
        raw_reward = accuracy + calibration_penalty
        final_reward = float(max(0.0, min(1.0, raw_reward)))

        return Reward(
            value=round(final_reward, 4),
            accuracy_reward=round(accuracy, 4),
            calibration_penalty=round(calibration_penalty, 4),
            actual_accuracy=round(accuracy, 4),
        )

    def get_history(self) -> Dict[str, list]:
        return {
            "predictions": self._predictions,
            "hidden_truths": self._hidden_truths,
            "confidences": self._confidences,
        }
