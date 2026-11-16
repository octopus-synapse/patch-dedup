"""Confidence-filtered NLI consensus; pseudo labels never claim human review."""

import numpy as np


def consensus_relation(large, small):
    large, small = np.asarray(large), np.asarray(small)
    if (large.max(-1) < 0.98).any() or not np.array_equal(large.argmax(-1), small.argmax(-1)):
        return None
    pair = tuple(large.argmax(-1).tolist())
    return {
        (0, 0): ("redundant", False),
        (1, 1): ("distinct", False),
        (2, 2): ("contradictory", False),
        (1, 0): ("partial_overlap", False),
        (0, 1): ("partial_overlap", True),
    }.get(pair)
