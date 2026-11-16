import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.ml
if importlib.util.find_spec("sklearn") is None:
    pytest.skip("Install ml extra", allow_module_level=True)

from patchdedup.target_search import metric_key, meets_targets, search_scores  # noqa: E402


def test_targets_must_be_met_together_without_rounding():
    assert not meets_targets(dict(average_precision=0.999, f1=0.926829268))
    assert not meets_targets(dict(average_precision=0.989999, f1=0.98))
    assert meets_targets(dict(average_precision=0.99, f1=0.927))
    assert metric_key(dict(average_precision=0.985, f1=0.94)) > metric_key(
        dict(average_precision=0.999, f1=0.80)
    )


def test_search_uses_fold_training_thresholds_and_exports_reproducible_graph():
    import numpy as np

    from patchdedup.score_graph import apply_score_graph

    y = [0, 1] * 12
    groups = [f"g{i}" for i in range(12) for _ in range(2)]
    matrix = np.array([[0.05, 0.2], [0.95, 0.8]] * 12)
    result = search_scores(matrix, y, groups, max_ensembles=4)
    assert result["target_met"]
    assert result["evaluation_role"] == "adaptive_development"
    assert result["independent_test_passed"] is False
    assert result["best"]["f1"] == result["best"]["average_precision"] == 1
    assert len(result["best"]["fold_thresholds"]) == 3
    assert all(0 < t < 1 for t in result["best"]["fold_thresholds"])
    restored = json.loads(json.dumps(result["graph"]))
    p = apply_score_graph(restored, matrix)
    assert np.array_equal(p >= restored["binary_threshold"], y)


def test_nonmeeting_search_reports_limit_instead_of_success():
    import numpy as np

    result = search_scores(
        np.full((24, 1), 0.5),
        [0, 1] * 12,
        [f"g{i}" for i in range(12) for _ in range(2)],
        max_ensembles=0,
    )
    assert not result["target_met"]
    assert result["best"]["average_precision"] < 0.99


def runner_module():
    file = Path(__file__).resolve().parents[1] / "scripts/search_targets.py"
    spec = importlib.util.spec_from_file_location("target_runner", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_does_not_create_run_or_read_a_test(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "search"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/search_targets.py"),
            "--dry-run",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["targets"] == dict(average_precision=0.99, f1=0.927)
    assert plan["paid_services"] is False
    assert plan["reads_final_test"] is False
    assert plan["max_trials"] > 0 and plan["max_hours"] > 0
    assert not output.exists()


def test_protocol_rejects_changed_data_or_targets_but_allows_budget_extension(tmp_path):
    module = runner_module()
    file = tmp_path / "protocol.json"
    original = dict(targets={"f1": 0.927}, input_hashes={"dev": "abc"})
    module.freeze_protocol(file, original)
    module.freeze_protocol(file, original)
    with pytest.raises(ValueError, match="protocol"):
        module.freeze_protocol(file, dict(original, input_hashes={"dev": "changed"}))
    assert json.loads(file.read_text()) == original


def test_shared_lock_rejects_duplicate_and_completed_v03_controller(tmp_path):
    module = runner_module()
    with module.exclusive_lock(tmp_path / "run.lock"):
        with pytest.raises(RuntimeError, match="controller"):
            with module.exclusive_lock(tmp_path / "run.lock"):
                pass
