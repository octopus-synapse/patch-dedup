"""Stage explicit model/dataset files; never uploads anything automatically."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from patchdedup.dataset import save_json
from patchdedup.encoder import verify_bundle, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/extended-42/selected-model"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/publication"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Publication staging already exists; choose a new directory")
    verify_bundle(args.model)
    model = args.output / "model"
    shutil.copytree(args.model, model)
    shutil.copy("hub/model/README.md", model / "README.md")
    shutil.copy("licenses/MINILM-APACHE-2.0.txt", model / "LICENSE")
    shutil.copy("licenses/NOTICE.md", model / "NOTICE.md")
    write_manifest(model)
    dataset = args.output / "dataset"
    for subset in ("initial", "refined", "challenge", "holdout_v02"):
        (dataset / subset).mkdir(parents=True)
    for split in ("train", "validation", "test"):
        src = Path("data/v01") / f"{split}.jsonl"
        shutil.copy(src, dataset / "initial" / src.name)
        target = dataset / "refined" / src.name
        target.write_bytes(
            src.read_bytes()
            + (Path("data/refinement/train.jsonl").read_bytes() if split == "train" else b"")
        )
    shutil.copy("data/challenge/test.jsonl", dataset / "challenge/test.jsonl")
    shutil.copy("data/v02/test.jsonl", dataset / "holdout_v02/test.jsonl")
    for name in ("v01", "refinement", "challenge", "v02"):
        (dataset / "manifests").mkdir(exist_ok=True)
        shutil.copy(Path("data") / name / "freeze.json", dataset / "manifests" / f"{name}.json")
    shutil.copytree("data/raw", dataset / "source")
    shutil.copy("hub/dataset/README.md", dataset / "README.md")
    shutil.copy("licenses/DATA-CC-BY-4.0.txt", dataset / "LICENSE")
    save_json(
        dataset / "provenance.json",
        dict(
            author="Enzo Ferracini Patti with Codex",
            reviewer="same authoring AI",
            human_reviewed=False,
            license="CC-BY-4.0",
            code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        ),
    )
    for version in ("v01", "v02"):
        destination = model / "evaluation" / version
        destination.mkdir(parents=True)
        for filename in ("test.json", "selection.json", "uncertainty.json"):
            src = Path("reports") / version / filename
            if src.exists():
                shutil.copy(src, destination / filename)
    write_manifest(model)
    verify_bundle(model)
    print(json.dumps({"model": str(model), "dataset": str(dataset)}))


if __name__ == "__main__":
    main()
