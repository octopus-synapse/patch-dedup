import hashlib
import json

import pytest

from patchdedup.dataset import prepare_dataset, verify_freeze


def test_build_is_deterministic_and_grouped(tmp_path):
    source = tmp_path / "source.psv"
    lines = []
    for i in range(20):
        lines.append(f"test|A {i}|P {i}|Q {i}|D {i}|E {i}|C {i}\n")
    source.write_text("".join(lines))
    out = tmp_path / "data"
    manifest = prepare_dataset(source, out, seed=42)
    assert manifest["groups"] == 20
    assert manifest["pairs"] == 160
    assert verify_freeze(out) == manifest
    for split in ("train", "validation", "test"):
        rows = [json.loads(s) for s in (out / f"{split}.jsonl").read_text().splitlines()]
        assert rows and all(r["split"] == split for r in rows)
    out2 = tmp_path / "other"
    prepare_dataset(source, out2, seed=42)
    for filename in manifest["sha256"]:
        assert (out / filename).read_bytes() == (out2 / filename).read_bytes()
    with pytest.raises(FileExistsError):
        prepare_dataset(source, out, seed=42)


def test_freeze_detects_tampering(tmp_path):
    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.jsonl").write_text("data")
    digest = hashlib.sha256(b"data").hexdigest()
    (tmp_path / "freeze.json").write_text(json.dumps({"sha256": {"train.jsonl": digest}}))
    (tmp_path / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        verify_freeze(tmp_path)


def test_refinement_is_training_only(tmp_path):
    source = tmp_path / "ref.psv"
    source.write_text("test|A|P|Q|D|E|C\n")
    out = tmp_path / "refinement"
    result = prepare_dataset(source, out, train_only=True)
    assert set(result["sha256"]) == {"train.jsonl"}
    assert not (out / "test.jsonl").exists()
    rows = [json.loads(s) for s in (out / "train.jsonl").read_text().splitlines()]
    assert all(r["split"] == "train" and r["scenario_id"].startswith("ref-") for r in rows)


def test_fresh_challenge_is_frozen_before_experiments(tmp_path):
    from patchdedup.dataset import prepare_challenge

    source = tmp_path / "challenge.psv"
    source.write_text("g1|test|A|B|redundant\ng1|test|A|C|distinct\n")
    out = tmp_path / "challenge"
    result = prepare_challenge(source, out)
    assert result["pairs"] == 2 and result["groups"] == 1
    assert verify_freeze(out) == result
    rows = [json.loads(s) for s in (out / "test.jsonl").read_text().splitlines()]
    assert rows[0]["text_a_id"] == rows[1]["text_a_id"]
    assert all(r["split"] == "test" for r in rows)
