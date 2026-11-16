import importlib.util
import json

import pytest

pytestmark = pytest.mark.ml
if importlib.util.find_spec("sklearn") is None:
    pytest.skip("Install the ml extra", allow_module_level=True)

from patchdedup.experiment import choose_candidate, final_evaluation  # noqa: E402


def test_selection_uses_validation_and_keeps_earliest_tie():
    candidates = [
        dict(name="initial", validation=dict(average_precision=0.8, f1=0.7)),
        dict(name="refined", validation=dict(average_precision=0.9, f1=0.6)),
        dict(name="tie", validation=dict(average_precision=0.9, f1=0.6)),
    ]
    assert choose_candidate(candidates)["name"] == "refined"


def test_final_evaluation_refuses_to_reuse_test(tmp_path):
    (tmp_path / "test.json").write_text(json.dumps({"frozen": True}))
    with pytest.raises(FileExistsError):
        final_evaluation({}, tmp_path / "absent", tmp_path, "cpu")
