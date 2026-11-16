"""Stage a mDeBERTa-root research bundle and all explicitly licensed dataset files."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from patchdedup.dataset import save_json, sha256
from patchdedup.encoder import verify_bundle, write_manifest
from patchdedup.v03_data import augmented_training

TOKEN_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "tokenizer.model",
    "cross_encoder.json",
}


def copy_model(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file() and path.name in TOKEN_FILES:
            shutil.copy2(path, destination / path.name)
    if not (destination / "model.safetensors").exists():
        raise ValueError("Missing safe checkpoint during export")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/v03/publication"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh publication staging directory")
    graph = json.loads(
        Path("artifacts/v03/calibrated-score-graph.json").read_text(encoding="utf-8")
    )
    research_revision = sha256("artifacts/v03/calibrated-score-graph.json")
    primary = next(
        (i for i, c in enumerate(graph["components"]) if c["initial_key"] == "nli_deberta"), 0
    )
    main_component = graph["components"][primary]
    if main_component["initial_key"] != "nli_deberta":
        raise RuntimeError(
            "Selected graph has no mDeBERTa component; review model naming before publishing"
        )
    root = args.output / "model"
    copy_model(main_component["directory"], root)
    for i, component in enumerate(graph["components"]):
        source = Path(component["directory"])
        if (
            component.get("model_sha256")
            and sha256(source / "model.safetensors") != component["model_sha256"]
        ):
            raise ValueError("Selected component was modified")
        relative = "." if i == primary else f"components/{i}"
        if i != primary:
            copy_model(source, root / relative)
        component["path"] = relative
        component["source_directory"] = component.pop("directory")
    save_json(root / "score_graph.json", graph)
    embedding = Path("artifacts/extended-42/selected-model")
    shutil.copytree(embedding, root / "embeddings")
    for name in ("manifest.json", "patchdedup.json", "thresholds.json"):
        (root / "embeddings" / name).unlink(missing_ok=True)
    shutil.copy("licenses/MINILM-APACHE-2.0.txt", root / "embeddings/LICENSE")
    save_json(root / "thresholds.json", graph["conformal"]["thresholds"])
    save_json(
        root / "patchdedup.json",
        dict(
            model_version="0.3.0-experimental",
            model_revision="sha256:" + sha256(root / "model.safetensors"),
            embedding_model_revision="sha256:" + sha256(root / "embeddings/model.safetensors"),
            embedding_directory="embeddings",
            score_pipeline=True,
            decision_source="ensemble" if len(graph["components"]) > 1 else "cross_encoder",
            decision_policy_version="0.3.0",
            research_graph_sha256=research_revision,
            calibration_pairs=graph["conformal"]["calibration_pairs"],
            primary_component=main_component["name"],
        ),
    )
    shutil.copy("hub/mdeberta/README.md", root / "README.md")
    shutil.copy("licenses/NLI-MIT.txt", root / "LICENSE")
    shutil.copy("licenses/NLI-NOTICE.md", root / "NOTICE.md")
    shutil.copy("artifacts/nli-deberta/README.md", root / "upstream-model-card.md")
    evaluation = root / "evaluation"
    evaluation.mkdir()
    for file in Path("reports/v03").glob("*.json"):
        shutil.copy(file, evaluation / file.name)
    write_manifest(root)
    verify_bundle(root)
    dataset = args.output / "dataset"
    if Path("artifacts/publication/dataset").exists():
        shutil.copytree("artifacts/publication/dataset", dataset)
    else:
        raise FileNotFoundError(
            "Historical dataset staging absent; restore the v02 dataset before export"
        )
    (dataset / "v03").mkdir()
    rows = augmented_training()
    (dataset / "v03/train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    (dataset / "v03/validation.jsonl").write_bytes(
        Path("data/v01/validation.jsonl").read_bytes()
        + Path("data/v03/development.jsonl").read_bytes()
    )
    for role in ("calibration", "test"):
        shutil.copy(Path("data/v03") / f"{role}.jsonl", dataset / "v03" / f"{role}.jsonl")
    shutil.copy("data/v03/freeze.json", dataset / "manifests/v03.json")
    shutil.copy("data/raw/development-v03.psv", dataset / "source/development-v03.psv")
    if Path("artifacts/v03/pseudo-train.jsonl").exists():
        shutil.copy("artifacts/v03/pseudo-train.jsonl", dataset / "v03/pseudo_train.jsonl")
    shutil.copy("hub/dataset/README.md", dataset / "README.md")
    save_json(
        dataset / "manifests/v03_training.json",
        dict(
            sha256={
                f"v03/{role}.jsonl": sha256(dataset / "v03" / f"{role}.jsonl")
                for role in ("train", "validation", "calibration", "test")
            },
            counts={"train": 744, "validation": 132, "calibration": 20, "test": 40},
        ),
    )
    save_json(
        dataset / "provenance.json",
        dict(
            author="Enzo Ferracini Patti with Codex",
            reviewer="same authoring AI",
            human_reviewed=False,
            license="CC-BY-4.0",
            export_code_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            protocol_sha256=sha256("configs/experiment-v03.json"),
            data_manifest="manifests/v03_training.json",
            note="Pseudo labels are unreviewed and excluded from the default training split",
        ),
    )
    save_json(
        args.output / "export.json",
        dict(
            model=str(root),
            dataset=str(dataset),
            research_graph_sha256=research_revision,
            model_weight_sha256=sha256(root / "model.safetensors"),
        ),
    )
    print(json.dumps(dict(model=str(root), dataset=str(dataset))))


if __name__ == "__main__":
    main()
