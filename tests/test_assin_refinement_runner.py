import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def runner():
    file = Path(__file__).resolve().parents[1] / "scripts/refine_assin.py"
    spec = importlib.util.spec_from_file_location("refine_assin", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_is_bounded_and_does_not_create_output(tmp_path):
    module = runner()
    out = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(module.ROOT / "scripts/refine_assin.py"),
            "--dry-run",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["max_candidates"] == 6 and plan["max_hours"] == 12
    assert plan["test_used_for_selection"] is False
    assert plan["paid_services"] is False and plan["publishes"] is False
    assert not out.exists()


def test_freeze_rejects_changed_protocol_and_writes_utf8(tmp_path):
    module = runner()
    file = tmp_path / "protocol.json"
    value = {"descrição": "paráfrase"}
    module.freeze(file, value)
    module.freeze(file, value)
    assert json.loads(file.read_text(encoding="utf-8")) == value
    with pytest.raises(ValueError):
        module.freeze(file, {"changed": True})


def test_selection_data_loading_does_not_read_test(monkeypatch):
    module = runner()
    original = Path.read_text

    def protected(path, *args, **kwargs):
        if path.name in ("test.jsonl", "predictions.json", "comparison.json"):
            raise AssertionError("Final test accessed before selection")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", protected)
    train, validation, professional = module.load_inputs()
    assert train and validation and professional
    assert all(r["source"] == "nilc-nlp/assin" for r in validation)
    assert all(r.get("source") != "nilc-nlp/assin" for r in professional)
    assert not ({r["scenario_id"] for r in train} & {r["scenario_id"] for r in validation})
