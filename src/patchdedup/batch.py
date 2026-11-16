"""Bounded JSONL inference; one verified model load, no input text in output errors."""

import json

from pydantic import ValidationError

from patchdedup.schemas import CompareInput

MAX_LINE_BYTES = 65536


def error(code, message):
    return {"error": {"code": code, "message": message}}


def comparison_error(exc):
    if isinstance(exc, ImportError):
        return "ML_DEPENDENCIES_UNAVAILABLE"
    if isinstance(exc, ValidationError):
        code = exc.errors(include_input=False)[0]["type"]
        return code if code in {"EMPTY_TEXT", "INPUT_TOO_LONG", "INVALID_UTF8"} else "INVALID_INPUT"
    code = str(exc).split(":", 1)[0]
    return code if code in {"INPUT_TOO_LONG", "INVALID_SCORE"} else "MODEL_UNAVAILABLE"


def iter_lines(stream):
    line = 0
    while raw := stream.readline(MAX_LINE_BYTES + 1):
        line += 1
        oversized = len(raw) > MAX_LINE_BYTES
        if oversized:
            while raw and not raw.endswith(b"\n"):
                raw = stream.readline(MAX_LINE_BYTES + 1)
        yield line, None if oversized else raw


def compare_stream(stream, factory, output, errors):
    try:
        comparator = factory()
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(json.dumps(error(comparison_error(exc), "Cannot load local model.")), file=errors)
        return 2
    failed = False
    records = 0
    try:
        for line, raw in iter_lines(stream):
            if raw is not None and not raw.strip():
                continue
            records += 1
            code = None
            if raw is None:
                code = "INPUT_TOO_LONG"
            else:
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    code = "INVALID_UTF8"
            if code is None:
                try:
                    request = CompareInput.model_validate_json(text)
                except ValidationError as exc:
                    code = comparison_error(exc)
                    if exc.errors(include_input=False)[0]["type"] == "json_invalid":
                        code = "INVALID_JSON"
            if code is None:
                try:
                    result = comparator.compare(request)
                except (ValueError, OSError, RuntimeError) as exc:
                    code = comparison_error(exc)
            if code is not None:
                failed = True
                record = dict(line=line, **error(code, "Comparison failed."))
            else:
                record = dict(line=line, result=result.model_dump(mode="json"))
            print(json.dumps(record, allow_nan=False), file=output, flush=True)
    except OSError:
        print(json.dumps(error("INPUT_UNAVAILABLE", "Cannot read input.")), file=errors)
        return 2
    if not records:
        print(json.dumps(error("EMPTY_INPUT", "No records to compare.")), file=errors)
        return 2
    return int(failed)
