"""Validation-only development followed by a single frozen final comparison."""

import argparse
import gc
import json
import shutil
from pathlib import Path

from patchdedup.data import audit_file
from patchdedup.dataset import save_json, sha256, verify_freeze
from patchdedup.decision import decide
from patchdedup.encoder import SentenceEncoder
from patchdedup.evaluation import evaluate_scores, select_policy
from patchdedup.schemas import Thresholds
from patchdedup.training import environment, seed_all, train_candidate


def load_rows(path):
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]


def choose_candidate(candidates):
    return max(
        candidates, key=lambda c: (c["validation"]["average_precision"], c["validation"]["f1"])
    )


def lexical_scores(train, rows):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = sorted({r[key] for r in train for key in ("text_a", "text_b")})
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=True).fit(texts)
    a = vectorizer.transform([r["text_a"] for r in rows])
    b = vectorizer.transform([r["text_b"] for r in rows])
    return np.clip(np.asarray(a.multiply(b).sum(axis=1)).ravel(), -1, 1).tolist()


def score_checkpoint(directory, rows, device):
    import torch

    encoder = SentenceEncoder(directory, device)
    scores = encoder.score_pairs(rows)
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def development(config, output, reports, device):
    output, reports = Path(output), Path(reports)
    if output.exists() or (reports / "selection.json").exists():
        raise FileExistsError("Choose fresh development and report directories")
    output.mkdir(parents=True)
    reports.mkdir(parents=True, exist_ok=True)
    initial_freeze = verify_freeze(config["dataset"])
    refinement_freeze = verify_freeze(config["refinement_dataset"])
    train = load_rows(Path(config["dataset"]) / "train.jsonl")
    validation = load_rows(Path(config["dataset"]) / "validation.jsonl")
    extra = load_rows(Path(config["refinement_dataset"]) / "train.jsonl")
    # Structural checks may inspect test identities; no model scores test here.
    combined = output / "combined-data.jsonl"
    combined.write_bytes(
        (Path(config["dataset"]) / "all.jsonl").read_bytes()
        + (Path(config["refinement_dataset"]) / "train.jsonl").read_bytes()
    )
    audit = audit_file(combined)
    save_json(reports / "data-audit.json", audit)
    if not audit["valid"]:
        raise ValueError("Combined dataset audit failed")
    seed_all(config["seed"])
    targets = [r["binary_label"] for r in validation]
    candidates = []

    def add(name, scores, directory=None, training=None):
        policy = select_policy(targets, scores)
        candidate = dict(
            name=name,
            directory=str(directory) if directory else None,
            policy=policy,
            validation=evaluate_scores(targets, scores, policy),
        )
        if training:
            candidate["training"] = training
        candidates.append(candidate)
        save_json(
            reports / f"{name}-validation.json",
            dict(
                candidate=candidate,
                predictions=[
                    dict(pair_id=r["pair_id"], label=r["binary_label"], score=s)
                    for r, s in zip(validation, scores, strict=True)
                ],
            ),
        )
        print(json.dumps({"candidate": name, "validation": candidate["validation"]}), flush=True)

    add("lexical", lexical_scores(train, validation))
    add(
        "base",
        score_checkpoint(config["base_directory"], validation, device),
        config["base_directory"],
    )
    initial_config = dict(config, model_version="0.1.0-initial")
    report = train_candidate(
        config["base_directory"], train, validation, output / "initial", initial_config, device
    )
    initial_model = output / "initial" / "model"
    add("initial", score_checkpoint(initial_model, validation, device), initial_model, report)
    for loss in config["refinement_candidates"]:
        name = f"refined_{loss}"
        cfg = dict(
            config,
            loss=loss,
            learning_rate=config["refinement_learning_rate"],
            model_version=f"0.1.0-refined-{loss}",
        )
        report = train_candidate(
            initial_model, train + extra, validation, output / name, cfg, device
        )
        directory = output / name / "model"
        add(name, score_checkpoint(directory, validation, device), directory, report)
    best = choose_candidate([c for c in candidates if c["name"] not in ("base", "lexical")])
    selection = dict(
        chosen_finetuned=best["name"],
        candidates=candidates,
        config=config,
        dataset=initial_freeze,
        refinement_dataset=refinement_freeze,
        environment=environment(),
        test_evaluated=False,
        selection_rule=config["selection"],
    )
    shutil.copytree(best["directory"], output / "selected-model")
    save_json(output / "selection.json", selection)
    save_json(reports / "selection.json", selection)
    print(json.dumps({"selected": best["name"], "test_evaluated": False}), flush=True)


def final_evaluation(config, output, reports, device):
    output, reports = Path(output), Path(reports)
    if (reports / "test.json").exists() or (output / "test-started.json").exists():
        raise FileExistsError("Test has already been opened; do not reuse for selection")
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    if config != selection["config"]:
        raise ValueError("Configuration changed after selection")
    if verify_freeze(config["dataset"]) != selection["dataset"]:
        raise ValueError("Dataset changed after selection")
    if verify_freeze(config["refinement_dataset"]) != selection["refinement_dataset"]:
        raise ValueError("Refinement dataset changed after selection")
    # Exclusive marker precedes the first test read and survives interrupted evaluations.
    with (output / "test-started.json").open("x") as marker:
        marker.write(json.dumps({"selection_sha256": sha256(output / "selection.json")}))
    rows = load_rows(Path(config["dataset"]) / "test.jsonl")
    train = load_rows(Path(config["dataset"]) / "train.jsonl")
    targets = [r["binary_label"] for r in rows]
    results = {}
    selected_errors = []
    for candidate in selection["candidates"]:
        name = candidate["name"]
        scores = (
            lexical_scores(train, rows)
            if name == "lexical"
            else score_checkpoint(candidate["directory"], rows, device)
        )
        policy = candidate["policy"]
        metrics = evaluate_scores(targets, scores, policy)
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
            if name == selection["chosen_finetuned"] and pred["binary_prediction"] != pred["label"]:
                selected_errors.append(dict(pred, text_a=row["text_a"], text_b=row["text_b"]))
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
            reports / f"{name}-test-predictions.json",
            dict(policy=policy, metrics=metrics, predictions=predictions),
        )
    final = dict(
        methods=results,
        chosen_finetuned=selection["chosen_finetuned"],
        test_pairs=len(rows),
        test_groups=len({r["scenario_id"] for r in rows}),
        selection_sha256=sha256(output / "selection.json"),
        environment=environment(),
        limitation="Synthetic authored pilot; same AI generated and reviewed; no human validation",
    )
    save_json(reports / "test.json", final)
    save_json(reports / "selected-errors.json", {"errors": selected_errors})
    print(json.dumps(final), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["develop", "final"])
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/run-42"))
    parser.add_argument("--reports", type=Path, default=Path("reports/v01"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    (development if args.stage == "develop" else final_evaluation)(
        config, args.output, args.reports, args.device
    )


if __name__ == "__main__":
    main()
