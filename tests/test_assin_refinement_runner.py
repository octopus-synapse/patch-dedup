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


def test_candidate_lock_waits_then_rechecks_completion(tmp_path):
    import fcntl
    import time

    module = runner()
    lock = tmp_path / "candidate.lock"
    acquired = tmp_path / "acquired"
    with module.candidate_lock(lock):
        code = (
            "import fcntl,pathlib,sys; "
            'f=open(sys.argv[1],"a"); '
            "fcntl.flock(f,fcntl.LOCK_EX); "
            'pathlib.Path(sys.argv[2]).write_text("acquired")'
        )
        child = subprocess.Popen([sys.executable, "-c", code, str(lock), str(acquired)])
        try:
            time.sleep(0.1)
            assert child.poll() is None
            assert not acquired.exists()
            with lock.open("a") as stream:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            # The process must survive until the context releases the first lock.
            pass
    try:
        assert child.wait(timeout=5) == 0
        assert acquired.read_text() == "acquired"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_protocol_only_accepts_explicit_scheduling_amendment(tmp_path):
    import copy

    module = runner()
    path = tmp_path / "protocol.json"
    old = dict(
        inputs={"train": "frozen"},
        code_sha256={"scripts/refine_assin.py": "old", "trainer": "unchanged"},
    )
    new = copy.deepcopy(old)
    new["code_sha256"]["scripts/refine_assin.py"] = "new"
    module.freeze_run_protocol(path, old)
    with pytest.raises(ValueError):
        module.freeze_run_protocol(path, new)
    module.save(
        tmp_path / "scheduling-amendment.json",
        dict(
            previous_runner_sha256="old",
            current_runner_sha256="new",
            user_requested=True,
            candidate_configs_changed=False,
        ),
    )
    module.freeze_run_protocol(path, new)
    assert json.loads(path.read_text()) == old
    changed_data = dict(new, inputs={"train": "changed"})
    with pytest.raises(ValueError):
        module.freeze_run_protocol(path, changed_data)
    changed_trainer = copy.deepcopy(new)
    changed_trainer["code_sha256"]["trainer"] = "changed"
    with pytest.raises(ValueError):
        module.freeze_run_protocol(path, changed_trainer)


def test_worker_skips_completed_checkpoint_after_candidate_lock(tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = runner()
    calls = []
    args = SimpleNamespace(output=tmp_path, worker="last", device="cuda")
    monkeypatch.delenv("PATCHDEDUP_CUDA_MEMORY_FRACTION", raising=False)

    def train(_):
        calls.append("trained")
        completed = tmp_path / "candidates/last/run/training.json"
        completed.parent.mkdir(parents=True)
        completed.write_text("{}")

    monkeypatch.setattr(module, "run_worker", train)
    module.worker(args)
    module.worker(args)
    assert calls == ["trained"]


def test_parallel_worker_memory_limit_is_validated_before_training(tmp_path, monkeypatch):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    module = runner()
    calls = []
    args = SimpleNamespace(output=tmp_path, worker="last", device="cuda")
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", calls.append)
    monkeypatch.setattr(module, "run_worker", lambda _: calls.append("trained"))
    monkeypatch.setenv("PATCHDEDUP_CUDA_MEMORY_FRACTION", "0.45")
    module.worker(args)
    assert calls == [0.45, "trained"]
    monkeypatch.setenv("PATCHDEDUP_CUDA_MEMORY_FRACTION", "0.9")
    with pytest.raises(ValueError):
        module.worker(args)
