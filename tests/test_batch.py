"""Batch software contracts; fake comparators do not evaluate ML quality."""

import io
import json

from patchdedup.schemas import CompareInput


def batch_api():
    from patchdedup.batch import compare_stream

    return compare_stream


class Result:
    def model_dump(self, **kwargs):
        return {"decision": "review", "requires_review": True}


def test_batch_loads_once_and_keeps_physical_line_numbers():
    calls = []

    class Comparator:
        def compare(self, request):
            assert isinstance(request, CompareInput)
            calls.append(request.text_a)
            return Result()

    def factory():
        calls.append("loaded")
        return Comparator()

    stream = io.BytesIO(b'{"text_a":"A","text_b":"B"}\n\n{"text_a":"C","text_b":"D"}\n')
    output, errors = io.StringIO(), io.StringIO()
    code = batch_api()(stream, factory, output, errors)
    assert code == 0 and errors.getvalue() == ""
    assert calls == ["loaded", "A", "C"]
    assert [p["line"] for p in map(json.loads, output.getvalue().splitlines())] == [1, 3]


def test_invalid_records_do_not_abort_or_echo_private_text():
    stream = io.BytesIO(
        b'private invalid json\n\xff\n{"text_a":" ","text_b":"private"}\n'
        b'{"text_a":"valid","text_b":"valid"}\n'
    )
    output, errors = io.StringIO(), io.StringIO()
    comparator = type("Comparator", (), {"compare": lambda self, request: Result()})()
    assert batch_api()(stream, lambda: comparator, output, errors) == 1
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [r.get("error", {}).get("code") for r in rows] == [
        "INVALID_JSON",
        "INVALID_UTF8",
        "EMPTY_TEXT",
        None,
    ]
    assert "private" not in output.getvalue() + errors.getvalue()


def test_token_overflow_is_reported_without_truncation_or_stopping():
    class Comparator:
        def compare(self, request):
            if request.text_a == "too long":
                raise ValueError("INPUT_TOO_LONG: private detail")
            return Result()

    stream = io.BytesIO(b'{"text_a":"too long","text_b":"B"}\n{"text_a":"A","text_b":"B"}\n')
    output, errors = io.StringIO(), io.StringIO()
    assert batch_api()(stream, Comparator, output, errors) == 1
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert rows[0]["error"]["code"] == "INPUT_TOO_LONG"
    assert "result" in rows[1] and "private" not in output.getvalue()


def test_missing_bundle_is_a_single_safe_startup_error():
    def factory():
        raise ValueError("MODEL_UNAVAILABLE: private path")

    output, errors = io.StringIO(), io.StringIO()
    code = batch_api()(io.BytesIO(b'{"text_a":"A","text_b":"B"}\n'), factory, output, errors)
    assert code == 2 and output.getvalue() == ""
    assert json.loads(errors.getvalue())["error"]["code"] == "MODEL_UNAVAILABLE"
    assert "private" not in errors.getvalue()


def test_oversized_line_does_not_split_into_fake_records():
    output, errors = io.StringIO(), io.StringIO()
    stream = io.BytesIO(b"x" * 70000 + b'\n{"text_a":"A","text_b":"B"}\n')
    comparator = type("Comparator", (), {"compare": lambda self, request: Result()})()
    assert batch_api()(stream, lambda: comparator, output, errors) == 1
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(rows) == 2 and rows[0]["error"]["code"] == "INPUT_TOO_LONG"
    assert rows[1]["line"] == 2 and "result" in rows[1]


def test_empty_input_is_explicit_and_unsupported_language_is_not_relabelled():
    comparator = type("Comparator", (), {"compare": lambda self, request: Result()})()
    output, errors = io.StringIO(), io.StringIO()
    assert batch_api()(io.BytesIO(b"\n"), lambda: comparator, output, errors) == 2
    assert json.loads(errors.getvalue())["error"]["code"] == "EMPTY_INPUT"
    output, errors = io.StringIO(), io.StringIO()
    raw = b'{"text_a":"A","text_b":"B","language":"en"}\n'
    assert batch_api()(io.BytesIO(raw), lambda: comparator, output, errors) == 1
    assert json.loads(output.getvalue())["error"]["code"] == "INVALID_INPUT"
