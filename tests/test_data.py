import hashlib

import pytest

from patchdedup.data import audit_file


def codes(report):
    return {error["code"] for error in report["errors"]}


def test_valid_fixture_has_counts_and_manifest(records, write_data):
    path = write_data(records)
    report = audit_file(path)
    assert report["valid"] is True
    assert report["records"] == 6
    assert report["counts"]["splits"] == {"train": 2, "validation": 2, "test": 2}
    assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["counts"]["review_status"] == {"unreviewed": 6}
    assert report["warnings"]  # Fixtures are not reviewed benchmark data.


@pytest.mark.parametrize(
    "key,value,code",
    [
        ("scenario_id", "same-scenario", "SCENARIO_LEAKAGE"),
        ("text_a_id", "same-id", "TEXT_ID_LEAKAGE"),
        ("text_a", " Mesmo\ttexto. ", "TEXT_LEAKAGE"),
    ],
)
def test_cross_split_leakage(records, write_data, key, value, code):
    records[0][key] = value
    records[2][key] = value.strip().replace("\t", " ")
    assert code in codes(audit_file(write_data(records)))


def test_unicode_normalization_for_leakage(records, write_data):
    records[0]["text_a"] = "Não fiz."
    records[2]["text_b"] = "Na\u0303o   fiz."
    assert "TEXT_LEAKAGE" in codes(audit_file(write_data(records)))


def test_inverted_pair_is_duplicate(records, write_data):
    reverse = records[0] | dict(
        pair_id="reverse",
        text_a_id=records[0]["text_b_id"],
        text_b_id=records[0]["text_a_id"],
        text_a=records[0]["text_b"],
        text_b=records[0]["text_a"],
    )
    assert "DUPLICATE_PAIR" in codes(audit_file(write_data(records + [reverse])))


def test_duplicate_pair_with_fresh_ids(records, write_data):
    duplicate = records[0] | dict(pair_id="duplicate", text_a_id="fresh-a", text_b_id="fresh-b")
    assert "DUPLICATE_PAIR" in codes(audit_file(write_data(records + [duplicate])))


def test_duplicate_pair_id(records, write_data):
    records[1]["pair_id"] = records[0]["pair_id"]
    assert "DUPLICATE_PAIR_ID" in codes(audit_file(write_data(records)))


def test_text_id_has_consistent_content(records, write_data):
    records[1]["text_a_id"] = records[0]["text_a_id"]
    assert "TEXT_ID_CONFLICT" in codes(audit_file(write_data(records)))


def test_shared_scenario_within_split_is_allowed(records, write_data):
    records[1]["scenario_id"] = records[0]["scenario_id"]
    assert audit_file(write_data(records))["valid"]


def test_ambiguous_is_excluded_from_binary_counts(records, write_data):
    extra = records[0] | dict(
        pair_id="ambiguous",
        text_a_id="x",
        text_b_id="y",
        text_a="Liderei a API.",
        text_b="Participei da API.",
        relation="ambiguous",
        binary_label=None,
    )
    report = audit_file(write_data(records + [extra]))
    assert report["valid"]
    assert report["binary_records"] == 6
    assert report["counts"]["relations"]["ambiguous"] == 1


@pytest.mark.parametrize(
    "body,code",
    [
        ("", "EMPTY_DATASET"),
        ("{}\n", "INVALID_RECORD"),
        ("{oops}\n", "INVALID_JSON"),
        ("[]\n", "INVALID_RECORD"),
    ],
)
def test_malformed_files(tmp_path, body, code):
    path = tmp_path / "bad.jsonl"
    path.write_text(body)
    report = audit_file(path)
    assert not report["valid"]
    assert code in codes(report)
    assert "text_a" not in str(report["errors"])


def test_missing_split_and_class(records, write_data):
    report = audit_file(write_data(records[:1]))
    assert {"MISSING_SPLIT", "MISSING_CLASS"} <= codes(report)


def test_invalid_utf8(tmp_path):
    path = tmp_path / "bytes.jsonl"
    path.write_bytes(b"\xff\n")
    assert "INVALID_UTF8" in codes(audit_file(path))
