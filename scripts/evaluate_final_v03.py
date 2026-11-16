"""Single reserved evaluation after all v03 model/policy choices have been frozen."""

import argparse
import json
from pathlib import Path

from patchdedup.calibration import calibration_metrics, prediction_sets
from patchdedup.dataset import save_json, sha256, verify_freeze
from patchdedup.decision import decide
from patchdedup.evaluation import evaluate_scores
from patchdedup.experiment import load_rows, score_checkpoint
from patchdedup.research_scores import component_matrix
from patchdedup.schemas import Thresholds
from patchdedup.score_graph import apply_score_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/v03"))
    parser.add_argument("--reports", type=Path, default=Path("reports/v03"))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if (args.reports / "test.json").exists():
        raise FileExistsError("Final v03 test already evaluated")
    graph_path = args.output / "calibrated-score-graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    calibration = json.loads((args.reports / "calibration.json").read_text(encoding="utf-8"))
    if sha256(graph_path) != calibration["graph_sha256"]:
        raise ValueError("Calibrated graph changed")
    verify_freeze("data/v03")
    with (args.output / "test-started.json").open("x", encoding="utf-8") as marker:
        json.dump({"graph_sha256": sha256(graph_path)}, marker)
    rows = load_rows("data/v03/test.jsonl")
    y = [r["binary_label"] for r in rows]
    matrix = component_matrix(graph["components"], rows, args.device)
    scores = apply_score_graph(graph, matrix)
    policy = dict(
        binary_threshold=graph["binary_threshold"], thresholds=graph["conformal"]["thresholds"]
    )
    metrics = evaluate_scores(y, scores, policy)
    metrics.update(calibration_metrics(y, scores))
    sets = prediction_sets(scores, graph["conformal"]["q"])
    metrics.update(
        conformal_marginal_coverage=sum(
            label in choices for label, choices in zip(y, sets, strict=True)
        )
        / len(y),
        mean_prediction_set_size=sum(map(len, sets)) / len(sets),
        empty_set_rate=sum(not choices for choices in sets) / len(sets),
    )
    predictions = [
        dict(
            row,
            score=float(score),
            binary_prediction=int(score >= graph["binary_threshold"]),
            decision=decide(float(score), Thresholds(**policy["thresholds"])),
            prediction_set=choices,
        )
        for row, score, choices in zip(rows, scores, sets, strict=True)
    ]
    # Raw primary baseline is compared with its development-fixed policy, not tuned here.
    baseline = json.loads(
        (args.reports / "candidates/nli_deberta.json").read_text(encoding="utf-8")
    )
    base_scores = component_matrix(
        [dict(directory=baseline["directory"], aggregation="min")], rows, args.device
    )[:, 0]
    baseline_metrics = evaluate_scores(y, base_scores, baseline["policy"])
    legacy_selection = json.loads(Path("reports/v02/selection.json").read_text(encoding="utf-8"))
    legacy = next(
        c
        for c in legacy_selection["candidates"]
        if c["name"] == legacy_selection["chosen_finetuned"]
    )
    legacy_directory = Path(legacy["directory"])
    if not legacy_directory.exists():
        legacy_directory = Path("artifacts/extended-42/selected-model")
    legacy_scores = score_checkpoint(legacy_directory, rows, args.device)
    legacy_metrics = evaluate_scores(y, legacy_scores, legacy["policy"])
    save_json(
        args.reports / "test.json",
        dict(
            metrics=metrics,
            primary_pretrained_baseline=baseline_metrics,
            previous_v02_embedding_baseline=legacy_metrics,
            test_pairs=len(rows),
            test_groups=len({r["scenario_id"] for r in rows}),
            graph_sha256=sha256(graph_path),
            components=graph["components"],
            policy=policy,
            limitation="Same AI author; not human-independent or representative of actual resumes",
        ),
    )
    save_json(args.reports / "test-predictions.json", dict(predictions=predictions))
    save_json(
        args.reports / "selected-errors.json",
        dict(errors=[p for p in predictions if p["binary_prediction"] != p["binary_label"]]),
    )
    print(json.dumps(dict(selected=metrics, baseline=baseline_metrics)))


if __name__ == "__main__":
    main()
