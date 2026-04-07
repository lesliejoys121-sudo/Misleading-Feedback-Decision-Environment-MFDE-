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
        calibration_error = abs(conf - accuracy)
        total_accuracy += accuracy
        total_calibration += calibration_error

    n = len(predictions)
    avg_accuracy = total_accuracy / n
    avg_calibration_error = total_calibration / n

    # Score: average accuracy minus calibration error, clamped to [0, 1]
    score = avg_accuracy - avg_calibration_error
    return float(max(0.0, min(1.0, score)))
