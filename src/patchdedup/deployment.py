"""Prepare an allowlisted model context for an offline Docker image."""

import argparse
import shutil
from pathlib import Path

from patchdedup.encoder import verify_bundle


def prepare_checkpoint(source, output):
    source, output = Path(source), Path(output)
    manifest = verify_bundle(source)
    output.mkdir(parents=True, exist_ok=False)
    for name in [*manifest, "manifest.json"]:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
        destination.chmod(0o644)
    verify_bundle(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare_checkpoint(args.source, args.output)


if __name__ == "__main__":
    main()
