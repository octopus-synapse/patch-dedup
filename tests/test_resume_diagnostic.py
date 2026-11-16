"""Software diagnostics; fixtures do not measure any model's quality."""

import importlib.util
from pathlib import Path

import pytest


def diagnostic():
    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_resume_pilot.py"
    spec = importlib.util.spec_from_file_location("diagnose_resume_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inconclusive_and_rejected_pairs_are_not_false_positives_or_correct_negatives():
    rows = [
        dict(status="scored", target=0, binary_prediction=1, decision="redundant"),
        dict(status="scored", target=0, binary_prediction=0, decision="not_redundant"),
        dict(status="scored", target=None, binary_prediction=1, decision="redundant"),
        dict(status="input_too_long", target=0),
    ]
    result = diagnostic().summarize(rows)
    assert result["input_count"] == 4
    assert result["scored_count"] == 3
    assert result["scored_negative_count"] == 2
    assert result["binary_false_positive_count"] == 1
    assert result["binary_false_positive_rate"] == 0.5
    assert result["inconclusive_decisions"] == {"redundant": 1}
    assert result["rejected_count"] == 1
    assert "accuracy" not in result and "f1" not in result and "average_precision" not in result


def test_binary_false_positive_and_review_are_separate():
    rows = [dict(status="scored", target=0, binary_prediction=1, decision="review")]
    result = diagnostic().summarize(rows)
    assert result["binary_false_positive_count"] == 1
    assert result["negative_decisions"] == {"review": 1}
    assert diagnostic().summarize([])["binary_false_positive_rate"] is None


def test_context_required_and_non_development_inputs_are_rejected():
    module = diagnostic()
    pair = dict(
        pair_id="fixture",
        split="development_pilot_only",
        contextual_binary_target=0,
        left=dict(text="Same words", model_text="Company A: Same words"),
        right=dict(text="Same words", model_text="Company B: Same words"),
    )
    assert module.model_row(pair) == dict(
        text_a="Company A: Same words", text_b="Company B: Same words"
    )
    with pytest.raises(ValueError, match="development"):
        module.model_row(dict(pair, split="test"))
    del pair["left"]["model_text"]
    with pytest.raises(ValueError, match="context"):
        module.model_row(pair)


def test_review_csv_preserves_unicode_under_ascii_locale(tmp_path):
    import os
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts/diagnose_resume_pilot.py"
    code = r"""
import importlib.util
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("diagnostic", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
predictions = [dict(pair_id="p", status="scored", target=0, relation="distinct")]
pairs = [dict(pair_id="p", left=dict(model_text="a" + chr(231)),
              right=dict(model_text="b"), rationale_ptbr="n" + chr(227) + "o")]
m.write_review(Path(sys.argv[2]), predictions, pairs)
"""
    env = dict(os.environ, LC_ALL="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
    result = subprocess.run(
        [sys.executable, "-c", code, str(script), str(tmp_path)], env=env, capture_output=True
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert "aç" in (tmp_path / "review.csv").read_text(encoding="utf-8")
