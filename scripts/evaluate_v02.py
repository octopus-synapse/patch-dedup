"""Open the fresh test once, after selection; never train or select from it."""

import argparse
import gc
import json
from pathlib import Path

import torch

from patchdedup.cross_encoder import CrossScorer
from patchdedup.dataset import save_json, sha256, verify_freeze
from patchdedup.decision import decide
from patchdedup.evaluation import evaluate_scores
from patchdedup.experiment import lexical_scores, load_rows, score_checkpoint
from patchdedup.schemas import Thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/extended-42"))
    parser.add_argument("--reports", type=Path, default=Path("reports/v02"))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if (args.output / "test-started.json").exists() or (args.reports / "test.json").exists():
        raise FileExistsError("Fresh test already opened; no reuse for model selection")
    selection = json.loads((args.output / "selection.json").read_text())
    config = selection["config"]
    if verify_freeze(config["fresh_holdout"]) != selection["fresh_holdout"]:
        raise ValueError("Fresh test changed after selection")
    with (args.output / "test-started.json").open("x") as marker:
        marker.write(json.dumps({"selection_sha256": sha256(args.output / "selection.json")}))
    rows = load_rows(Path(config["fresh_holdout"]) / "test.jsonl")
    train = load_rows(Path(config["dataset"]) / "train.jsonl")
    train += load_rows(Path(config["refinement_dataset"]) / "train.jsonl")
    y = [r["binary_label"] for r in rows]
    results = {}
    errors = []
    for candidate in selection["candidates"]:
        name = candidate["name"]
        if candidate["kind"] == "lexical":
            scores = lexical_scores(train, rows)
        elif candidate["kind"] == "cross_encoder":
            scorer = CrossScorer(candidate["directory"], args.device)
            scores = scorer.score_pairs(rows)
            del scorer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            scores = score_checkpoint(candidate["directory"], rows, args.device)
        policy = candidate["policy"]
        metrics = evaluate_scores(y, scores, policy)
        predictions = []
        for row, score in zip(rows, scores, strict=True):
            pred = dict(
                pair_id=row["pair_id"],
                scenario_id=row["scenario_id"],
                relation=row["relation"],
                label=row["binary_label"],
                score=score,
                binary_prediction=int(score >= policy["binary_threshold"]),
                decision=decide(score, Thresholds(**policy["thresholds"])),
            )
            predictions.append(pred)
            if name == selection["chosen_finetuned"] and pred["label"] != pred["binary_prediction"]:
                errors.append(dict(pred, text_a=row["text_a"], text_b=row["text_b"]))
        negative = [p for p in predictions if p["label"] == 0]
        metrics["hard_negative_false_positive_rate"] = sum(
            p["binary_prediction"] for p in negative
        ) / len(negative)
        metrics["relation_errors"] = {
            relation: dict(
                support=sum(p["relation"] == relation for p in predictions),
                errors=sum(
                    p["relation"] == relation and p["binary_prediction"] != p["label"]
                    for p in predictions
                ),
            )
            for relation in sorted({p["relation"] for p in predictions})
        }
        results[name] = metrics
        save_json(
            args.reports / f"{name}-test-predictions.json",
            dict(policy=policy, metrics=metrics, predictions=predictions),
        )
    save_json(
        args.reports / "test.json",
        dict(
            methods=results,
            chosen_finetuned=selection["chosen_finetuned"],
            chosen_bi_encoder=selection["chosen_bi_encoder"],
            test_pairs=len(rows),
            test_groups=len({r["scenario_id"] for r in rows}),
            selection_sha256=sha256(args.output / "selection.json"),
            limitation="Fresh authored pilot; same AI author; previous diagnostics influenced general design",
        ),
    )
    save_json(args.reports / "selected-errors.json", dict(errors=errors))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
