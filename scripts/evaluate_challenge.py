"""Evaluate a separately frozen challenge after all development selection."""

import argparse
import json
from pathlib import Path

from patchdedup.dataset import save_json, sha256, verify_freeze
from patchdedup.decision import decide
from patchdedup.evaluation import evaluate_scores
from patchdedup.experiment import lexical_scores, load_rows, score_checkpoint
from patchdedup.schemas import Thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/run-42"))
    parser.add_argument("--reports", type=Path, default=Path("reports/v01"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    destination = args.reports / "challenge.json"
    if destination.exists():
        raise FileExistsError("Challenge already evaluated; do not retune on it")
    selection = json.loads((args.output / "selection.json").read_text(encoding="utf-8"))
    freeze = verify_freeze("data/challenge")
    rows = load_rows("data/challenge/test.jsonl")
    train = load_rows(Path(selection["config"]["dataset"]) / "train.jsonl")
    results = {}
    for candidate in selection["candidates"]:
        scores = (
            lexical_scores(train, rows)
            if candidate["name"] == "lexical"
            else score_checkpoint(candidate["directory"], rows, args.device)
        )
        binary_rows = [
            (r, s) for r, s in zip(rows, scores, strict=True) if r["binary_label"] is not None
        ]
        metrics = evaluate_scores(
            [r["binary_label"] for r, _ in binary_rows],
            [s for _, s in binary_rows],
            candidate["policy"],
        )
        thresholds = Thresholds(**candidate["policy"]["thresholds"])
        predictions = [
            dict(
                row,
                score=score,
                decision=decide(score, thresholds),
                binary_prediction=int(score >= candidate["policy"]["binary_threshold"]),
            )
            for row, score in zip(rows, scores, strict=True)
        ]
        ambiguous = [p for p in predictions if p["relation"] == "ambiguous"]
        results[candidate["name"]] = dict(
            metrics=metrics,
            ambiguous_support=len(ambiguous),
            ambiguous_sent_to_review=sum(p["decision"] == "review" for p in ambiguous),
            predictions=predictions,
        )
    save_json(
        destination,
        dict(
            methods=results,
            freeze=freeze,
            chosen_finetuned=selection["chosen_finetuned"],
            selection_sha256=sha256(args.output / "selection.json"),
            note="No challenge-based selection; same AI author, separately written constructions",
        ),
    )
    print(json.dumps({name: value["metrics"] for name, value in results.items()}))


if __name__ == "__main__":
    main()
