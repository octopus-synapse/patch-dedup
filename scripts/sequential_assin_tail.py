"""Temporarily suspend the auxiliary sixth worker until the fifth exits, preserving progress."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import sys
import time


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return fields[19], args
    except (FileNotFoundError, ProcessLookupError):
        return None


def save(output, stage, **values):
    value = dict(
        stage=stage,
        execution_mode="sequential",
        updated=dt.datetime.now(dt.timezone.utc).isoformat(),
        **values,
    )
    target = output / "sequential-status.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps(value), flush=True)


def sequentialize(output, pause_pid, wait_pid, poll_interval=5):
    pause_identity, wait_identity = identity(pause_pid), identity(wait_pid)
    if pause_identity is None:
        save(output, "sixth_already_exited")
        return
    if b"binary-seed42" not in pause_identity[1]:
        raise ValueError("Refusing to suspend a process other than the sixth candidate")
    if wait_identity is not None and b"assin-three-way" not in wait_identity[1]:
        raise ValueError("The preceding process must be the fifth candidate")
    try:
        if wait_identity is not None:
            os.kill(pause_pid, signal.SIGSTOP)
            save(
                output,
                "sixth_paused_for_fifth",
                paused_pid=pause_pid,
                active_pid=wait_pid,
                progress_preserved_in_memory=True,
            )
            while identity(wait_pid) == wait_identity and identity(pause_pid) == pause_identity:
                time.sleep(poll_interval)
    finally:
        # Validate process start time and command, so PID reuse cannot target an unrelated process.
        if identity(pause_pid) == pause_identity:
            os.kill(pause_pid, signal.SIGCONT)
            save(output, "sixth_resumed", active_pid=pause_pid, progress_preserved_in_memory=True)
        else:
            save(output, "sixth_already_exited")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pause-pid", type=int, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sequentialize(args.output.resolve(), args.pause_pid, args.wait_pid)
