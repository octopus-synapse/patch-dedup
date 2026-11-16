"""Fresh grouped development/calibration/test data, frozen before v03 selection."""

import hashlib
import json
import random
from pathlib import Path

from patchdedup.dataset import save_json, sha256
from patchdedup.schemas import PairRecord


def build_protocol_data(source, output):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh dataset directory")
    lines = source.read_text(encoding="utf-8").splitlines()
    if len(lines) != 60:
        raise ValueError("Expected sixty authored groups for the preregistered split")
    order = list(range(len(lines)))
    random.Random(3137).shuffle(order)
    roles = {
        i: ("development" if rank < 30 else "calibration" if rank < 40 else "test")
        for rank, i in enumerate(order)
    }
    rows = {role: [] for role in ("development", "calibration", "test")}
    for index, line in enumerate(lines):
        group, domain, a, p, negative, relation = line.split("|")
        scenario = "v03-" + group
        role = roles[index]
        for j, (b, label) in enumerate(((p, "redundant"), (negative, relation))):
            record = PairRecord(
                pair_id=f"{scenario}-{j}",
                scenario_id=scenario,
                text_a_id=scenario + "-" + hashlib.sha256(a.encode()).hexdigest()[:12],
                text_b_id=scenario + "-" + hashlib.sha256(b.encode()).hexdigest()[:12],
                text_a=a,
                text_b=b,
                relation=label,
                binary_label=int(label == "redundant"),
                difficulty="fresh_authored_v03",
                domain=domain,
                source_type="synthetic",
                review_status="reviewed",
                annotation_version="0.3.0",
                split="test" if role == "test" else "validation",
            )
            rows[role].append(record.model_dump())
    output.mkdir(parents=True, exist_ok=True)
    for role, records in rows.items():
        (output / f"{role}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        )
    manifest = dict(
        groups_by_role={"development": 30, "calibration": 10, "test": 20},
        pairs_by_role={role: len(records) for role, records in rows.items()},
        seed=3137,
        sha256={f"{role}.jsonl": sha256(output / f"{role}.jsonl") for role in rows},
        source_sha256=sha256(source),
        provenance={
            "author": "same authoring AI",
            "human_reviewed": False,
            "license": "CC-BY-4.0",
            "limitation": "Style still synthetic; prior diagnostics motivated categories, not final labels or selection",
        },
    )
    save_json(output / "freeze.json", manifest)
    return manifest


def augmented_training(root="data"):
    """Add omitted within-scenario combinations with the same authored relations."""
    root = Path(root)
    rows = [
        json.loads(line)
        for file in (root / "v01/train.jsonl", root / "refinement/train.jsonl")
        for line in file.read_text().splitlines()
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scenario_id"], []).append(row)
    additions = []
    for scenario, records in grouped.items():
        texts = {}
        for row in records:
            texts[row["text_a_id"].rsplit("-", 1)[1]] = row["text_a"]
            texts[row["text_b_id"].rsplit("-", 1)[1]] = row["text_b"]
        for j, (a, b, relation) in enumerate(
            (
                ("p", "c", "contradictory"),
                ("q", "c", "contradictory"),
                ("a", "e", "partial_overlap"),
                ("p", "d", "distinct"),
            )
        ):
            value = dict(
                records[0],
                pair_id=f"{scenario}-aug-{j}",
                text_a_id=f"{scenario}-{a}",
                text_b_id=f"{scenario}-{b}",
                text_a=texts[a],
                text_b=texts[b],
                relation=relation,
                binary_label=0,
            )
            additions.append(PairRecord(**value).model_dump())
    return rows + additions
