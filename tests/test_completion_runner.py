import importlib.util
import subprocess
import sys
from pathlib import Path


def runner_module():
    file = Path(__file__).resolve().parents[1] / "scripts/finish_remaining.py"
    spec = importlib.util.spec_from_file_location("completion_runner", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lock_prevents_duplicate_controller(tmp_path):
    module = runner_module()
    lock = tmp_path / "runner.lock"
    with module.exclusive_lock(lock):
        code = (
            'import fcntl,sys; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
        )
        result = subprocess.run([sys.executable, "-c", code, str(lock)], capture_output=True)
        assert result.returncode != 0
    with module.exclusive_lock(lock):
        pass


def test_partial_archive_preserves_completed_checkpoints(tmp_path, monkeypatch):
    module = runner_module()
    monkeypatch.setattr(module, "RUN", tmp_path)
    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "training.json").write_text("{}")
    assert module.archive_partial(complete, "training.json") is None
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "checkpoint").write_bytes(b"preserved")
    archived = module.archive_partial(partial, "training.json")
    assert not partial.exists()
    assert (archived / "checkpoint").read_bytes() == b"preserved"


def test_dry_run_only_reports_plan():
    module = runner_module()
    before = (
        (module.CONTROL / "status.json").read_bytes()
        if (module.CONTROL / "status.json").exists()
        else None
    )
    result = subprocess.run(
        [
            sys.executable,
            str(module.ROOT / "scripts/finish_remaining.py"),
            "--dry-run",
            "--publish",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"paid_services": false' in result.stdout
    after = (
        (module.CONTROL / "status.json").read_bytes()
        if (module.CONTROL / "status.json").exists()
        else None
    )
    assert before == after
