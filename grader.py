from typing import List


def grade(
    predictions: List[float],
    hidden_truths: List[float],
    confidences: List[float],
) -> float:
    """
    Deterministic grader.
    Returns a score in [0.0, 1.0] based on:
      - Average prediction accuracy
      - Average confidence calibration
    """
    if not predictions:
        return 0.0

    total_accuracy = 0.0
    total_calibration = 0.0

    for pred, truth, conf in zip(predictions, hidden_truths, confidences):
        accuracy = max(0.0, 1.0 - abs(pred - truth))
        calibration = max(0.0, 1.0 - abs(conf - accuracy))
        total_accuracy += accuracy
        total_calibration += calibration

    n = len(predictions)
    avg_accuracy = total_accuracy / n
    avg_calibration = total_calibration / n

    # Score: 0.6 accuracy + 0.4 calibration, clamped to [0, 1]
    score = 0.6 * avg_accuracy + 0.4 * avg_calibration
    return float(max(0.0, min(1.0, score)))
