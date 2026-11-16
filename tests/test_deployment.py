"""Packaging integrity fixtures, not trained model weights."""

import hashlib
import json

import pytest


def fixture_bundle(path):
    path.mkdir()
    files = {
        "model.safetensors": b"software fixture, not real weights",
        "patchdedup.json": b"{}",
        "thresholds.json": b"{}",
        "modules.json": b"[]",
    }
    for name, data in files.items():
        (path / name).write_bytes(data)
    (path / "manifest.json").write_text(
        json.dumps({name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    )
    return path


def test_packaging_includes_only_verified_manifest_files(tmp_path):
    from patchdedup.deployment import prepare_checkpoint

    source = fixture_bundle(tmp_path / "source")
    (source / ".env").write_text("PRIVATE_TOKEN=do-not-copy")
    (source / "private.txt").write_text("do-not-copy")
    output = tmp_path / "output"
    prepare_checkpoint(source, output)
    assert {p.name for p in output.iterdir()} == {
        "manifest.json",
        "model.safetensors",
        "patchdedup.json",
        "thresholds.json",
        "modules.json",
    }
    for p in output.iterdir():
        assert p.read_bytes() == (source / p.name).read_bytes()
    with pytest.raises(FileExistsError):
        prepare_checkpoint(source, output)


def test_corrupt_checkpoint_cannot_be_packaged(tmp_path):
    from patchdedup.deployment import prepare_checkpoint

    source = fixture_bundle(tmp_path / "source")
    (source / "model.safetensors").write_bytes(b"tampered")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="MODEL_UNAVAILABLE"):
        prepare_checkpoint(source, output)
    assert not output.exists()
