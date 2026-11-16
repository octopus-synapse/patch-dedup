import importlib.util
import os
from pathlib import Path

import pytest

from patchdedup.encoder import Comparator, verify_bundle
from patchdedup.schemas import CompareInput


def test_missing_model_never_falls_back(tmp_path):
    with pytest.raises(ValueError, match="MODEL_UNAVAILABLE"):
        Comparator(tmp_path / "missing")


def test_corrupt_bundle_is_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text('{"weights.safetensors":"wrong"}')
    (tmp_path / "weights.safetensors").write_text("changed")
    with pytest.raises(ValueError, match="MODEL_UNAVAILABLE"):
        verify_bundle(tmp_path)


@pytest.mark.ml
@pytest.mark.real_model
@pytest.mark.skipif(not os.environ.get("PATCHDEDUP_MODEL"), reason="Set PATCHDEDUP_MODEL")
def test_real_checkpoint_offline_symmetry_and_limits():
    assert importlib.util.find_spec("sentence_transformers")
    model = Comparator(Path(os.environ["PATCHDEDUP_MODEL"]))
    a, b = "Automatizei relatórios com Python.", "Usei Python para automatizar relatórios."
    forward = model.compare(CompareInput(text_a=a, text_b=b))
    reverse = model.compare(CompareInput(text_a=b, text_b=a))
    assert abs(forward.similarity.score - reverse.similarity.score) < 1e-6
    assert forward.decision == reverse.decision
    assert -1 <= forward.similarity.score <= 1
    assert model.compare(CompareInput(text_a=a, text_b=a)).similarity.score > 0.999
    with pytest.raises(ValueError, match="INPUT_TOO_LONG"):
        model.compare(CompareInput(text_a="palavra " * 150, text_b=b))
