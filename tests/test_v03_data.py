import json

from patchdedup.v03_data import build_protocol_data


def test_v03_groups_are_disjoint_and_augmentation_keeps_train_groups(tmp_path):
    source = tmp_path / "source.psv"
    source.write_text("".join(f"g{i}|test|A {i}|P {i}|D {i}|distinct\n" for i in range(60)))
    out = tmp_path / "v03"
    result = build_protocol_data(source, out)
    assert result["groups_by_role"] == {"development": 30, "calibration": 10, "test": 20}
    seen = set()
    for role in ("development", "calibration", "test"):
        rows = [json.loads(s) for s in (out / f"{role}.jsonl").read_text().splitlines()]
        groups = {r["scenario_id"] for r in rows}
        assert not (groups & seen)
        seen |= groups
    assert len(seen) == 60


def test_training_data_is_utf8_even_after_process_locale_changes():
    import subprocess
    import sys

    code = "import locale; locale.setlocale(locale.LC_CTYPE, 'C'); from patchdedup.v03_data import augmented_training; print(len(augmented_training()))"
    result = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "744"
