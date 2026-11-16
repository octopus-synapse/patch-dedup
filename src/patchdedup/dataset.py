"""Deterministic grouped dataset construction and freeze verification."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from patchdedup.data import audit_file
from patchdedup.schemas import PairRecord


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: str | Path, value: dict) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def verify_freeze(directory: str | Path) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    for name, digest in manifest["sha256"].items():
        if Path(name).name != name or sha256(directory / name) != digest:
            raise ValueError("HASH_MISMATCH: frozen dataset differs from manifest")
    return manifest


def prepare_dataset(
    source: str | Path, output: str | Path, seed: int = 42, train_only: bool = False
) -> dict:
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new directory; frozen datasets cannot be overwritten")
    lines = [
        line.split("|") for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines or any(len(parts) != 7 for parts in lines):
        raise ValueError("Expected domain and six authored variants per scenario")
    order = list(range(len(lines)))
    random.Random(seed).shuffle(order)
    ntrain = int(len(lines) * 0.7)
    nval = max(1, int(len(lines) * 0.15))
    splits = {
        index: ("train" if rank < ntrain else "validation" if rank < ntrain + nval else "test")
        for rank, index in enumerate(order)
    }
    pairs = [
        ("a", "p", "redundant"),
        ("p", "q", "redundant"),
        ("a", "q", "redundant"),
        ("a", "d", "distinct"),
        ("p", "e", "partial_overlap"),
        ("a", "c", "contradictory"),
        ("q", "d", "distinct"),
        ("q", "e", "partial_overlap"),
    ]
    records = []
    group_counts = Counter()
    for index, (domain, *variants) in enumerate(lines):
        scenario = f"{'ref' if train_only else 'v01'}-{index:03}"
        split = "train" if train_only else splits[index]
        group_counts[split] += 1
        texts = dict(zip(("a", "p", "q", "d", "e", "c"), variants, strict=True))
        for pair_index, (left, right, relation) in enumerate(pairs):
            record = PairRecord(
                pair_id=f"{scenario}-p{pair_index}",
                scenario_id=scenario,
                text_a_id=f"{scenario}-{left}",
                text_b_id=f"{scenario}-{right}",
                text_a=texts[left],
                text_b=texts[right],
                relation=relation,
                binary_label=int(relation == "redundant"),
                difficulty=("paraphrase" if relation == "redundant" else "hard_negative"),
                domain=domain,
                source_type="synthetic",
                review_status="reviewed",
                annotation_version="0.1.0",
                split=split,
            )
            records.append(record.model_dump())
    output.mkdir(parents=True, exist_ok=True)
    names = ["train"] if train_only else ["train", "validation", "test"]
    for split in names:
        (output / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n" for r in records if r["split"] == split
            ),
            encoding="utf-8",
        )
    if not train_only:
        all_path = output / "all.jsonl"
        all_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        )
        audit = audit_file(all_path)
        save_json(output / "audit.json", audit)
        if not audit["valid"]:
            raise ValueError(f"Dataset audit failed: {audit['errors']}")
    files = [f"{split}.jsonl" for split in names]
    if not train_only:
        files.append("all.jsonl")
    manifest = dict(
        schema_version="0.1.0",
        seed=seed,
        groups=len(lines),
        pairs=len(records),
        groups_by_split=dict(group_counts),
        source_sha256=sha256(source),
        sha256={name: sha256(output / name) for name in files},
        provenance={
            "source": source.name,
            "author": "Codex authoring AI",
            "reviewer": "same authoring AI; no independent or human review",
            "human_reviewed": False,
            "license": "CC-BY-4.0",
            "limitation": "Shared authored constructions; not representative of real CVs",
        },
    )
    save_json(output / "freeze.json", manifest)
    return manifest


def prepare_challenge(source: str | Path, output: str | Path) -> dict:
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a fresh challenge directory")
    rows = []
    indices = Counter()
    for line in source.read_text(encoding="utf-8").splitlines():
        group, domain, a, b, relation = line.split("|")
        scenario = "holdout-v02-" + group
        record = PairRecord(
            pair_id=f"{scenario}-{indices[group]}",
            scenario_id=scenario,
            text_a_id=scenario + "-" + hashlib.sha256(a.encode()).hexdigest()[:12],
            text_b_id=scenario + "-" + hashlib.sha256(b.encode()).hexdigest()[:12],
            text_a=a,
            text_b=b,
            relation=relation,
            binary_label=None if relation == "ambiguous" else int(relation == "redundant"),
            difficulty="fresh_authored_holdout",
            domain=domain,
            source_type="synthetic",
            review_status="reviewed",
            annotation_version="0.2.0",
            split="test",
        )
        rows.append(record.model_dump())
        indices[group] += 1
    output.mkdir(parents=True, exist_ok=True)
    target = output / "test.jsonl"
    target.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    manifest = dict(
        groups=len(indices),
        pairs=len(rows),
        sha256={"test.jsonl": sha256(target)},
        source_sha256=sha256(source),
        provenance={
            "author": "same authoring AI",
            "human_reviewed": False,
            "license": "CC-BY-4.0",
            "purpose": "Fresh scenarios frozen before v02 selection; old tests are diagnostic only",
            "limitation": "General design informed by earlier aggregate errors; not independent human authorship",
        },
    )
    save_json(output / "freeze.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.source, args.output, train_only=args.train_only)))


if __name__ == "__main__":
    main()
