"""Offline data validation. No encoder is downloaded or simulated."""

import argparse
import json
import sys
from contextlib import nullcontext

from patchdedup import __version__
from pydantic import ValidationError

from patchdedup.data import audit_file
from patchdedup.schemas import CompareInput


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse messages can include arbitrary user input; keep errors private.
        print(
            json.dumps(
                {"error": {"code": "USAGE_ERROR", "message": "Invalid arguments; use --help."}}
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    parser = Parser(prog="patchdedup", description="PatchDedup experimental data tooling")
    parser.add_argument("--version", action="version", version=f"patchdedup {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser(
        "validate-data", help="Audit a JSONL file containing all three splits"
    )
    audit.add_argument("--input", required=True, help="UTF-8 JSONL path")
    compare = commands.add_parser("compare", help="Compare two texts using a verified local bundle")
    compare.add_argument("--model", required=True)
    compare.add_argument("--text-a", required=True)
    compare.add_argument("--text-b", required=True)
    compare.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    batch = commands.add_parser("batch", help="Compare JSONL pairs loading the local model once")
    batch.add_argument("--model", required=True)
    batch.add_argument("--input", required=True, help="UTF-8 JSONL file, or - for stdin")
    batch.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    if args.command == "batch":
        from patchdedup.batch import compare_stream, error

        def factory():
            from patchdedup.encoder import Comparator

            return Comparator(args.model, args.device)

        try:
            source = nullcontext(sys.stdin.buffer) if args.input == "-" else open(args.input, "rb")
            with source as stream:
                return compare_stream(stream, factory, sys.stdout, sys.stderr)
        except OSError:
            print(json.dumps(error("INPUT_UNAVAILABLE", "Cannot read input.")), file=sys.stderr)
            return 2
    if args.command == "compare":
        try:
            from patchdedup.encoder import Comparator

            request = CompareInput(text_a=args.text_a, text_b=args.text_b)
            result = Comparator(args.model, args.device).compare(request)
            print(result.model_dump_json())
            return 0
        except ValidationError as exc:
            code = exc.errors(include_input=False)[0]["type"]
            if code not in {"EMPTY_TEXT", "INPUT_TOO_LONG", "INVALID_UTF8"}:
                code = "INVALID_INPUT"
        except ImportError:
            code = "ML_DEPENDENCIES_UNAVAILABLE"
        except ValueError as exc:
            code = str(exc).split(":", 1)[0]
            if code not in {"MODEL_UNAVAILABLE", "INPUT_TOO_LONG", "INVALID_SCORE"}:
                code = "MODEL_UNAVAILABLE"
        print(
            json.dumps({"error": {"code": code, "message": "Comparison failed."}}), file=sys.stderr
        )
        return 2
    try:
        report = audit_file(args.input)
    except OSError:
        print(
            json.dumps(
                {"error": {"code": "INPUT_UNAVAILABLE", "message": "Cannot read input file."}}
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report["valid"] else 1
