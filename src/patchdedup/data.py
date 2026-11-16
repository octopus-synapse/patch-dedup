"""Audit exact identities and split integrity; never infer semantic equivalence."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from patchdedup.schemas import PairRecord


def audit_file(path: str | Path) -> dict:
    """Return a JSON-safe manifest without echoing potentially sensitive texts.

    A file contains all three splits. Each pair belongs to one scenario; callers
    must allocate scenarios before constructing pairs. Near-duplicate/template
    leakage requires a separate review and is not certified by this audit.
    """
    errors: list[dict] = []
    counts = {name: Counter() for name in ("splits", "relations", "review_status")}
    classes = {split: set() for split in ("train", "validation", "test")}
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    seen_text_pairs: set[tuple[str, str]] = set()
    identities: dict[str, dict[str, str]] = {
        "SCENARIO_LEAKAGE": {},
        "TEXT_ID_LEAKAGE": {},
        "TEXT_LEAKAGE": {},
    }
    text_contents: dict[str, str] = {}
    total = binary = 0
    digest = hashlib.sha256()

    def error(code: str, line: int | None = None) -> None:
        item = {"code": code}
        if line is not None:
            item["line"] = line
        if item not in errors:
            errors.append(item)

    def track(code: str, key: str, split: str, line: int) -> None:
        previous = identities[code].setdefault(key, split)
        if previous != split:
            error(code, line)

    with Path(path).open("rb") as stream:
        for line, raw in enumerate(stream, start=1):
            digest.update(raw)
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError:
                error("INVALID_UTF8", line)
                continue
            if not decoded.strip():
                continue
            try:
                payload = json.loads(decoded)
            except (ValueError, RecursionError):
                error("INVALID_JSON", line)
                continue
            try:
                record = PairRecord.model_validate(payload)
            except ValidationError:
                error("INVALID_RECORD", line)
                continue
            total += 1
            counts["splits"][record.split] += 1
            counts["relations"][record.relation] += 1
            counts["review_status"][record.review_status] += 1
            if record.binary_label is not None:
                binary += 1
                classes[record.split].add(record.binary_label)
            if record.pair_id in seen_ids:
                error("DUPLICATE_PAIR_ID", line)
            seen_ids.add(record.pair_id)
            pair = tuple(sorted((record.text_a_id, record.text_b_id)))
            text_pair = tuple(sorted((record.text_a, record.text_b)))
            if pair in seen_pairs or text_pair in seen_text_pairs:
                error("DUPLICATE_PAIR", line)
            seen_pairs.add(pair)
            seen_text_pairs.add(text_pair)
            track("SCENARIO_LEAKAGE", record.scenario_id, record.split, line)
            for text_id, text in (
                (record.text_a_id, record.text_a),
                (record.text_b_id, record.text_b),
            ):
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                track("TEXT_ID_LEAKAGE", text_id, record.split, line)
                track("TEXT_LEAKAGE", text_hash, record.split, line)
                previous = text_contents.setdefault(text_id, text_hash)
                if previous != text_hash:
                    error("TEXT_ID_CONFLICT", line)
    if not total:
        error("EMPTY_DATASET")
    for split, labels in classes.items():
        if not counts["splits"][split]:
            errors.append({"code": "MISSING_SPLIT", "split": split})
        elif labels != {0, 1}:
            errors.append({"code": "MISSING_CLASS", "split": split})
    warnings = ["NEAR_DUPLICATES_AND_TEMPLATE_FAMILIES_REQUIRE_REVIEW"]
    if counts["review_status"]["unreviewed"]:
        warnings.append("UNREVIEWED_RECORDS")
    return {
        "schema_version": "0.1.0",
        "valid": not errors,
        "sha256": digest.hexdigest(),
        "records": total,
        "binary_records": binary,
        "counts": {name: dict(sorted(value.items())) for name, value in counts.items()},
        "errors": errors,
        "warnings": warnings,
    }
