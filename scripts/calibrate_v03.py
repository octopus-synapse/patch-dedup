"""Fit uncertainty on reserved calibration groups after score graph selection."""

import argparse
import json
from pathlib import Path

from patchdedup.calibration import calibration_metrics, conformal_policy, prediction_sets
from patchdedup.dataset import save_json, sha256, verify_freeze
from patchdedup.experiment import load_rows
from patchdedup.research_scores import component_matrix
from patchdedup.score_graph import apply_score_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/v03"))
    parser.add_argument("--reports", type=Path, default=Path("reports/v03"))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if (args.reports / "calibration.json").exists():
        raise FileExistsError(
            "Reserved calibration already consumed; never use it for model search"
        )
    config = json.loads(Path("configs/experiment-v03.json").read_text(encoding="utf-8"))
    verify_freeze("data/v03")
    selected_path = args.output / "selected-score-graph.json"
    graph = json.loads(selected_path.read_text(encoding="utf-8"))
    selection = json.loads((args.reports / "selection.json").read_text(encoding="utf-8"))
    if sha256(selected_path) != selection["graph_sha256"]:
        raise ValueError("Score graph changed after selection")
    with (args.output / "calibration-started.json").open("x", encoding="utf-8") as file:
        json.dump({"selection_sha256": sha256(selected_path)}, file)
    rows = load_rows(config["calibration"])
    labels = [r["binary_label"] for r in rows]
    scores = apply_score_graph(graph, component_matrix(graph["components"], rows, args.device))
    conformal = conformal_policy(labels, scores, alpha=config["conformal_alpha"])
    graph["conformal"] = conformal
    graph["decision_policy_version"] = "0.3.0"
    save_json(args.output / "calibrated-score-graph.json", graph)
    sets = prediction_sets(scores, conformal["q"])
    save_json(
        args.reports / "calibration.json",
        dict(
            conformal=conformal,
            groups=len({r["scenario_id"] for r in rows}),
            metrics=calibration_metrics(labels, scores),
            graph_sha256=sha256(args.output / "calibrated-score-graph.json"),
            predictions=[
                dict(pair_id=r["pair_id"], label=y, score=float(s), prediction_set=choices)
                for r, y, s, choices in zip(rows, labels, scores, sets, strict=True)
            ],
            limitation="Only twenty calibration pairs; coverage assumes exchangeability and does not guarantee precision",
        ),
    )
    print(json.dumps(conformal))


if __name__ == "__main__":
    main()
