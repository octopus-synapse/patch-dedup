"""Attach the last ASSIN candidate to the existing controller without duplicate training."""

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/assin-refinement"
CHILD = None


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def locked(path):
    if not path.exists():
        return False
    with path.open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def ready(output):
    return (
        (output / "candidates/binary-hard-ranking/run/training.json").exists()
        and not (output / "candidates/binary-seed42/run/training.json").exists()
        and locked(output / "locks/assin-three-way.lock")
        and not locked(output / "locks/binary-seed42.lock")
    )


def tail_config(protocol, status):
    frozen = next(c for c in protocol["candidates"] if c["name"] == "binary-seed42")
    return dict(
        frozen,
        initial=protocol["initial"]["directory"],
        training=dict(frozen["training"], minimum_f1=status["incumbent_validation"]["f1"]),
    )


def gpu_free_mib():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def run(args):
    global CHILD
    output = args.output.resolve()
    protocol = read(output / "protocol.json")
    amendment = read(output / "scheduling-amendment.json")
    if not amendment.get("user_requested") or amendment.get("candidate_configs_changed"):
        raise ValueError("An explicit scheduling amendment is required")
    status_file = output / "parallel-status.json"
    started = time.monotonic()

    def update(stage, **values):
        status = dict(
            stage=stage,
            pid=os.getpid(),
            candidate="binary-seed42",
            updated=dt.datetime.now(dt.timezone.utc).isoformat(),
            **values,
        )
        save(status_file, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    with (output / "parallel-tail.lock").open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        update("waiting_for_last_two", memory_fraction=0.45)
        while time.monotonic() - started < args.max_wait_hours * 3600:
            state = read(output / "status.json")
            if (output / "comparison.json").exists() or state.get("stage") in (
                "complete",
                "failed",
                "interrupted",
            ):
                update("not_started", reason="Main controller ended; no duplicate launched")
                return
            if not Path(f"/proc/{state['pid']}").exists():
                update("not_started", reason="Main controller is no longer running")
                return
            if (output / "candidates/binary-seed42/run/training.json").exists():
                update("already_completed")
                return
            if ready(output):
                free = gpu_free_mib()
                if free >= 3277:
                    break
            time.sleep(10)
        else:
            update("sequential_fallback", reason="Parallel readiness/memory wait expired")
            return
        config = tail_config(protocol, state)
        path = output / "configs/binary-seed42.json"
        if path.exists() and read(path) != config:
            raise ValueError("Frozen final-candidate configuration changed")
        if not path.exists():
            save(path, config)
        env = dict(
            os.environ,
            PATCHDEDUP_CUDA_MEMORY_FRACTION="0.45",
            PYTHONUTF8="1",
            PYTHONUNBUFFERED="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        # Same worker and same frozen configuration as the original queue.
        # The worker lock makes the parent wait/skip, never launch a duplicate GPU model.
        update("parallel_training", free_mib=free, memory_fraction=0.45)
        try:
            with (output / "logs/binary-seed42.log").open("a", encoding="utf-8") as log:
                CHILD = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "scripts/refine_assin.py"),
                        "--output",
                        str(output),
                        "--worker",
                        "binary-seed42",
                        "--device",
                        "cuda",
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                code = CHILD.wait(timeout=3 * 3600)
            if code:
                update(
                    "sequential_fallback",
                    exit_code=code,
                    reason="Auxiliary worker failed; parent will retry alone after the fifth candidate",
                )
            else:
                update("parallel_completed")
        except BaseException:
            if CHILD is not None and CHILD.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(CHILD.pid, signal.SIGTERM)
                try:
                    CHILD.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(CHILD.pid, signal.SIGKILL)
                    CHILD.wait()
            update(
                "sequential_fallback",
                reason="Auxiliary worker interrupted; main queue remains active",
            )
            raise
        finally:
            CHILD = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-wait-hours", type=float, default=3)
    arguments = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    run(arguments)
