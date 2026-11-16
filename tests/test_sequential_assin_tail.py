import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


def runner():
    file = Path(__file__).resolve().parents[1] / "scripts/sequential_assin_tail.py"
    spec = importlib.util.spec_from_file_location("sequential_tail", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sequentialization_suspends_then_resumes_without_killing_worker(tmp_path):
    module = runner()
    wait = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(0.4)", "--worker", "assin-three-way"]
    )
    pause = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(10)", "--worker", "binary-seed42"]
    )
    thread = threading.Thread(
        target=lambda: module.sequentialize(tmp_path, pause.pid, wait.pid, poll_interval=0.02)
    )
    try:
        thread.start()
        deadline = time.monotonic() + 3
        while not (tmp_path / "sequential-status.json").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        status = json.loads((tmp_path / "sequential-status.json").read_text())
        assert status["stage"] == "sixth_paused_for_fifth"
        assert (
            Path(f"/proc/{pause.pid}/status")
            .read_text()
            .split("State:")[1]
            .lstrip()
            .startswith("T")
        )
        wait.wait(timeout=3)
        thread.join(timeout=3)
        assert not thread.is_alive()
        assert pause.poll() is None
        status = json.loads((tmp_path / "sequential-status.json").read_text())
        assert status["stage"] == "sixth_resumed"
        assert (
            not Path(f"/proc/{pause.pid}/status")
            .read_text()
            .split("State:")[1]
            .lstrip()
            .startswith("T")
        )
    finally:
        import os
        import signal

        for process in (pause, wait):
            if process.poll() is None:
                os.kill(process.pid, signal.SIGCONT)
                process.terminate()
            process.wait(timeout=3)
        thread.join(timeout=3)
