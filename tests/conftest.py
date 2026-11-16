import json

import pytest


@pytest.fixture
def records():
    values = []
    for split in ("train", "validation", "test"):
        for label in (0, 1):
            prefix = f"{split}-{label}"
            values.append(
                dict(
                    pair_id=prefix,
                    scenario_id=f"s-{prefix}",
                    text_a_id=f"{prefix}-a",
                    text_b_id=f"{prefix}-b",
                    text_a=f"Contribuição A {prefix}.",
                    text_b=f"Contribuição B {prefix}.",
                    relation="redundant" if label else "distinct",
                    binary_label=label,
                    difficulty="fixture",
                    domain="testing",
                    language="pt-BR",
                    source_type="synthetic",
                    review_status="unreviewed",
                    annotation_version="0.1.0",
                    split=split,
                )
            )
    return values


@pytest.fixture
def write_data(tmp_path):
    def write(records):
        path = tmp_path / "pairs.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        )
        return path

    return write
