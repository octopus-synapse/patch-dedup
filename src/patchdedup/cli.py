"""Offline data validation. No encoder is downloaded or simulated."""

import argparse
import json
import sys

from patchdedup import __version__
from patchdedup.data import audit_file


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
    args = parser.parse_args(argv)
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
