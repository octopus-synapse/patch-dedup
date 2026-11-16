import importlib.util

import pytest

pytestmark = pytest.mark.ml
if importlib.util.find_spec("sklearn") is None:
    pytest.skip("Install ml extra", allow_module_level=True)

from patchdedup.score_graph import crossfit_graph, fit_score_graph, apply_score_graph  # noqa: E402


def test_crossfit_selects_using_held_out_groups():
    y = [0, 1] * 12
    groups = [f"g{i}" for i in range(12) for _ in range(2)]
    matrix = [[0.1], [0.9]] * 12
    template = dict(kind="average", indices=[0], weights=[1.0], calibrator_kind="sigmoid")
    report = crossfit_graph(matrix, y, groups, template)
    assert report["macro_f1"] == 1
    assert len(report["oof_scores"]) == len(y)
    assert report["folds"] == 3


def test_stacking_graph_is_json_roundtrip_safe():
    import json
    import numpy as np

    x = [[0.1, 0.2], [0.2, 0.3], [0.8, 0.7], [0.9, 0.8]] * 3
    y = [0, 0, 1, 1] * 3
    template = dict(kind="stacking", indices=[0, 1], C=1.0, calibrator_kind="identity")
    graph = fit_score_graph(x, y, template)
    original = apply_score_graph(graph, x)
    restored = apply_score_graph(json.loads(json.dumps(graph)), x)
    assert np.allclose(original, restored)
    assert (restored >= 0).all() and (restored <= 1).all()
