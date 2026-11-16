import importlib.util

import pytest

pytestmark = pytest.mark.ml
if importlib.util.find_spec("sklearn") is None:
    pytest.skip("Install ml extra", allow_module_level=True)

from patchdedup.calibration import (  # noqa: E402
    apply_calibrator,
    fit_calibrator,
    group_folds,
    binary_threshold,
    conformal_policy,
    prediction_sets,
)  # noqa: E402


def test_group_folds_never_split_a_scenario():
    y = [0, 1] * 12
    groups = [f"g{i}" for i in range(12) for _ in range(2)]
    for train, valid in group_folds(y, groups):
        assert not ({groups[i] for i in train} & {groups[i] for i in valid})


@pytest.mark.parametrize("kind", ["identity", "sigmoid", "isotonic"])
def test_calibration_is_finite_and_serializable(kind):
    import json
    import numpy as np

    s = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    calibration = json.loads(json.dumps(fit_calibrator(kind, s, y)))
    values = apply_calibrator(calibration, s)
    assert np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_threshold_and_conformal_abstention_are_separate():
    assert 0.2 < binary_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) <= 0.8
    policy = conformal_policy([0, 1] * 10, [0.05, 0.95] * 10, alpha=0.1)
    sets = prediction_sets([0.01, 0.5, 0.99], policy["q"])
    assert sets == [[0], [], [1]]
    assert policy["thresholds"]["low"] < policy["thresholds"]["high"]


def test_tiny_calibration_defaults_to_review_not_false_confidence():
    policy = conformal_policy([0, 1], [0.1, 0.9], alpha=0.1)
    assert policy["q"] == 1
    assert policy["thresholds"]["positive_enabled"] is False
    assert prediction_sets([0, 0.5, 1], 1) == [[0, 1], [0, 1], [0, 1]]
