import json
from pathlib import Path

import pytest


@pytest.fixture
def records():
    path = Path(__file__).parent / "fixtures" / "pairs.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def write_data(tmp_path):
    def write(records):
        path = tmp_path / "pairs.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        )
        return path

    return write
