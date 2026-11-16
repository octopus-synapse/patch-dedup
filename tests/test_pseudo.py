import importlib.util

import pytest

pytestmark = pytest.mark.ml
if importlib.util.find_spec("numpy") is None:
    pytest.skip("Install ml extra", allow_module_level=True)

from patchdedup.pseudo import consensus_relation  # noqa: E402


def test_pseudo_labels_require_confidence_and_agreement():
    ent = [0.99, 0.005, 0.005]
    neu = [0.005, 0.99, 0.005]
    con = [0.005, 0.005, 0.99]
    assert consensus_relation([ent, ent], [ent, ent]) == ("redundant", False)
    assert consensus_relation([neu, ent], [neu, ent]) == ("partial_overlap", False)
    assert consensus_relation([ent, neu], [ent, neu]) == ("partial_overlap", True)
    assert consensus_relation([con, con], [con, con]) == ("contradictory", False)
    assert consensus_relation([ent, ent], [neu, neu]) is None
    assert consensus_relation([[0.5, 0.3, 0.2], ent], [ent, ent]) is None
