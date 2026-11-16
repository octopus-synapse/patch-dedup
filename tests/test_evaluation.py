import importlib.util

import pytest

pytestmark = pytest.mark.ml
if importlib.util.find_spec("sklearn") is None:
    pytest.skip("Install the ml extra", allow_module_level=True)

from patchdedup.evaluation import evaluate_scores, select_policy  # noqa: E402


def test_perfect_ranking_and_threshold_selection():
    targets = [0] * 6 + [1] * 6
    scores = [-0.2, 0, 0.1, 0.2, 0.3, 0.4, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    policy = select_policy(targets, scores)
    report = evaluate_scores(targets, scores, policy)
    assert report["f1"] == report["average_precision"] == 1
    assert report["confusion_matrix"] == [[6, 0], [0, 6]]
    assert report["coverage"] > 0
    assert report["positive_support"] == 6


def test_no_high_precision_threshold_disables_positive_decisions():
    targets = [1, 0] * 10
    policy = select_policy(targets, [0.5] * 20)
    assert policy["thresholds"]["positive_enabled"] is False
    report = evaluate_scores(targets, [0.5] * 20, policy)
    assert report["redundant_count"] == 0
    assert report["review_rate"] == 1


def test_review_remains_in_denominator():
    report = evaluate_scores(
        [0, 1, 0, 1],
        [0.1, 0.9, 0.5, 0.5],
        dict(binary_threshold=0.7, thresholds=dict(low=0.2, high=0.8)),
    )
    assert report["coverage"] == report["review_rate"] == 0.5
    assert report["review_count"] == 2


def test_invalid_evaluation_inputs():
    with pytest.raises(ValueError):
        select_policy([0], [0.1])
    with pytest.raises(ValueError):
        evaluate_scores([0, 1], [0.1], {})
