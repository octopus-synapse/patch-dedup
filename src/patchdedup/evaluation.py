"""Validation-only policy selection and binary/selective metrics."""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score
from sklearn.metrics import recall_score

from patchdedup.decision import decide
from patchdedup.schemas import Thresholds


def arrays(targets, scores):
    y, s = np.asarray(targets), np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.shape != y.shape or len(y) == 0:
        raise ValueError("Expected equally sized nonempty one-dimensional inputs")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all() or (abs(s) > 1).any():
        raise ValueError("Invalid labels or cosine scores")
    return y, s


def select_policy(targets, scores):
    """Freeze F1 binary cutoff and 90% precision/NPV selective boundaries.

    At least five decided validation pairs are required per enabled direction.
    This is a pilot heuristic, not a calibrated statistical safety guarantee.
    """
    y, s = arrays(targets, scores)
    if set(y) != {0, 1}:
        raise ValueError("Validation requires both classes")
    candidates = np.unique(np.concatenate(([-1.0, 1.0], s)))
    binary = max(
        candidates,
        key=lambda t: (
            f1_score(y, s >= t, zero_division=0),
            precision_score(y, s >= t, zero_division=0),
            -abs(t - 0.5),
        ),
    )
    positives = [float(t) for t in candidates if (s >= t).sum() >= 5 and y[s >= t].mean() >= 0.9]
    high = min(positives) if positives else 1.0
    negatives = [
        float(t)
        for t in candidates
        if t < high and (s < t).sum() >= 5 and (1 - y[s < t]).mean() >= 0.9
    ]
    low = max(negatives) if negatives else -1.0
    thresholds = Thresholds(
        low=low, high=high, positive_enabled=bool(positives), negative_enabled=bool(negatives)
    )
    return {
        "binary_threshold": float(binary),
        "thresholds": thresholds.model_dump(),
        "selection_split": "validation",
        "precision_target": 0.9,
        "negative_predictive_value_target": 0.9,
        "minimum_support": 5,
        "decision_policy_version": "0.1.0",
    }


def evaluate_scores(targets, scores, policy):
    y, s = arrays(targets, scores)
    predicted = s >= policy["binary_threshold"]
    thresholds = Thresholds(**policy["thresholds"])
    decisions = np.asarray([decide(float(score), thresholds) for score in s])
    positive = decisions == "redundant"
    negative = decisions == "not_redundant"
    review = decisions == "review"
    return {
        "support": len(y),
        "positive_support": int(y.sum()),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "macro_f1": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "average_precision": float(average_precision_score(y, s)) if y.sum() else None,
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "coverage": float((~review).mean()),
        "review_rate": float(review.mean()),
        "review_count": int(review.sum()),
        "redundant_count": int(positive.sum()),
        "not_redundant_count": int(negative.sum()),
        "selective_precision": float(y[positive].mean()) if positive.any() else None,
        "negative_predictive_value": float((1 - y[negative]).mean()) if negative.any() else None,
    }
