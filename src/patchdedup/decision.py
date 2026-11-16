"""Score policy independent of any encoder or training framework."""

import math

from patchdedup.schemas import Decision, Thresholds


def decide(score: float, thresholds: Thresholds) -> Decision:
    """Apply an explicit policy; cosine similarity is not a probability."""
    if not math.isfinite(score) or not -1 <= score <= 1:
        raise ValueError("INVALID_SCORE: expected finite cosine in [-1, 1]")
    if thresholds.positive_enabled and score >= thresholds.high:
        return "redundant"
    if thresholds.negative_enabled and score < thresholds.low:
        return "not_redundant"
    return "review"
