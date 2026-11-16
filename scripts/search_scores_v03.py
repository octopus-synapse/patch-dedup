"""Cross-fitted postprocessor/ensemble search using development groups only."""

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
from study_v03 import data

from patchdedup.calibration import binary_threshold
from patchdedup.dataset import save_json, sha256
from patchdedup.score_graph import crossfit_graph, fit_score_graph


def distributions(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(shifted)
    return p / p.sum(axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment-v03.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v03"))
    parser.add_argument("--reports", type=Path, default=Path("reports/v03"))
    args = parser.parse_args()
    if (args.reports / "selection.json").exists():
        raise FileExistsError("Score graph selection already frozen")
    config = json.loads(args.config.read_text())
    _, dev = data(config)
    y = [r["binary_label"] for r in dev]
    groups = [r["scenario_id"] for r in dev]
    components = []
    columns = []
    raw_candidates = sorted((args.reports / "candidates").glob("*.json"))
    for file in raw_candidates:
        candidate = json.loads(file.read_text())
        probabilities = distributions(np.load(candidate["cache"], allow_pickle=False))[:, :, 0]
        for aggregation in ("min", "mean", "geometric"):
            score = (
                probabilities.min(1)
                if aggregation == "min"
                else probabilities.mean(1)
                if aggregation == "mean"
                else np.sqrt(probabilities.prod(1))
            )
            components.append(
                dict(
                    name=candidate["name"] + "::" + aggregation,
                    directory=candidate["directory"],
                    initial_key=candidate["initial_key"],
                    aggregation=aggregation,
                    quantized=candidate.get("quantized", False),
                )
            )
            columns.append(score)
    matrix = np.asarray(columns).T
    outcomes = []

    def evaluate(template):
        result = crossfit_graph(matrix, y, groups, template)
        outcomes.append(result)
        return result

    for i in range(len(components)):
        for calibrator in config["calibrators"]:
            evaluate(dict(kind="average", indices=[i], weights=[1.0], calibrator_kind=calibrator))
    ranked = sorted(outcomes, key=lambda r: (r["macro_f1"], r["average_precision"]), reverse=True)
    pool = []
    seen = set()
    for row in ranked:
        index = row["template"]["indices"][0]
        component = components[index]
        key = (component["directory"], component["quantized"])
        if key not in seen:
            pool.append(index)
            seen.add(key)
        if len(pool) == 4:
            break
    for count in range(2, len(pool) + 1):
        for calibrator in config["calibrators"]:
            evaluate(
                dict(
                    kind="average",
                    indices=pool[:count],
                    weights=[1 / count] * count,
                    calibrator_kind=calibrator,
                )
            )
    for C in (0.1, 1.0, 10.0):
        for calibrator in config["calibrators"]:
            evaluate(dict(kind="stacking", indices=pool, C=C, calibrator_kind=calibrator))
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=8)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        weights = [max(1e-8, trial.suggest_float(f"w{i}", 0, 1)) for i in range(len(pool))]
        calibrator = trial.suggest_categorical("calibrator", config["calibrators"])
        report = evaluate(
            dict(kind="average", indices=pool, weights=weights, calibrator_kind=calibrator)
        )
        trial.set_user_attr("average_precision", report["average_precision"])
        return report["macro_f1"]

    study.optimize(objective, n_trials=config["ensemble_weight_trials"])
    best = max(outcomes, key=lambda r: (r["macro_f1"], r["average_precision"]))
    graph = fit_score_graph(
        matrix, y, best["template"], calibration_training_scores=best["raw_oof_scores"]
    )
    indices = graph["indices"]
    graph["components"] = [components[i] for i in indices]
    graph["indices"] = list(range(len(indices)))
    graph["binary_threshold"] = binary_threshold(y, best["oof_scores"])
    save_json(args.output / "selected-score-graph.json", graph)
    summary = dict(
        selected=best,
        components=graph["components"],
        graphs_evaluated=len(outcomes),
        development_pairs=len(dev),
        development_groups=len(set(groups)),
        config=config,
        model_epoch_selection_used_development=True,
        limitation="Cross-fitting controls postprocessor fitting, not all adaptive model selection bias",
        graph_sha256=sha256(args.output / "selected-score-graph.json"),
        test_opened=False,
        calibration_opened=False,
    )
    save_json(args.reports / "selection.json", summary)
    save_json(
        args.reports / "score-search.json",
        dict(
            components=components,
            results=outcomes,
            weight_trials=[
                dict(number=t.number, params=t.params, value=t.value) for t in study.trials
            ],
        ),
    )
    print(
        json.dumps(
            dict(
                selected_components=graph["components"],
                macro_f1=best["macro_f1"],
                average_precision=best["average_precision"],
                graphs=len(outcomes),
            )
        )
    )


if __name__ == "__main__":
    main()
