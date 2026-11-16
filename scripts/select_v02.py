"""Freeze the extended selection and export its verified inference bundle."""

import argparse
import json
import shutil
from pathlib import Path

from extend_experiment import data, record_candidate

from patchdedup.dataset import save_json, verify_freeze
from patchdedup.encoder import write_manifest
from patchdedup.experiment import choose_candidate, lexical_scores, score_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/extended-42"))
    parser.add_argument("--reports", type=Path, default=Path("reports/v02"))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment-v02.json"))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if (args.reports / "selection.json").exists():
        raise FileExistsError("Extended selection is already frozen")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train, val = data(config)
    bi = json.loads((args.reports / "bi-development.json").read_text(encoding="utf-8"))[
        "candidates"
    ]
    cross_data = json.loads((args.reports / "cross-development.json").read_text(encoding="utf-8"))
    cross = cross_data["candidates"]
    lexical = record_candidate(
        "lexical", "", "lexical", lexical_scores(train, val), val, args.reports
    )
    base = record_candidate(
        "base",
        config["base_directory"],
        "cosine",
        score_checkpoint(config["base_directory"], val, args.device),
        val,
        args.reports,
    )
    best_bi = choose_candidate(bi)
    best = choose_candidate(bi + cross)
    selected = args.output / "selected-model"
    shutil.copytree(best_bi["directory"], selected)
    metadata = json.loads((selected / "patchdedup.json").read_text(encoding="utf-8"))
    metadata.update(
        model_version="0.2.0-experimental",
        decision_source=("cross_encoder" if best["kind"] == "cross_encoder" else "cosine"),
        selected_embedding=best_bi["name"],
        selected_decision_model=best["name"],
    )
    if best["kind"] == "cross_encoder":
        shutil.copytree(best["directory"], selected / "reranker")
    save_json(selected / "thresholds.json", best["policy"]["thresholds"])
    save_json(selected / "patchdedup.json", metadata)
    write_manifest(selected)
    selection = dict(
        chosen_finetuned=best["name"],
        chosen_bi_encoder=best_bi["name"],
        candidates=[lexical, base] + bi + cross,
        skipped=cross_data["skipped"],
        config=config,
        fresh_holdout=verify_freeze(config["fresh_holdout"]),
        dataset=verify_freeze(config["dataset"]),
        refinement_dataset=verify_freeze(config["refinement_dataset"]),
        test_evaluated=False,
        selection_rule=config["selection"],
    )
    save_json(args.output / "selection.json", selection)
    save_json(args.reports / "selection.json", selection)
    print(
        json.dumps(
            {"embedding": best_bi["name"], "decision": best["name"], "test_evaluated": False}
        )
    )


if __name__ == "__main__":
    main()
