"""Predeclared extended experiments. The development stages never score a test."""

import argparse
import json
from pathlib import Path

from patchdedup.dataset import save_json, verify_freeze
from patchdedup.evaluation import evaluate_scores, select_policy
from patchdedup.experiment import load_rows, score_checkpoint
from patchdedup.training import train_candidate


def data(config):
    verify_freeze(config["dataset"])
    verify_freeze(config["refinement_dataset"])
    verify_freeze(config["fresh_holdout"])
    train = load_rows(Path(config["dataset"]) / "train.jsonl")
    train += load_rows(Path(config["refinement_dataset"]) / "train.jsonl")
    val = load_rows(Path(config["dataset"]) / "validation.jsonl")
    return train, val


def record_candidate(name, directory, kind, scores, val, reports):
    policy = select_policy([r["binary_label"] for r in val], scores)
    candidate = dict(
        name=name,
        directory=str(directory),
        kind=kind,
        policy=policy,
        validation=evaluate_scores([r["binary_label"] for r in val], scores, policy),
    )
    save_json(reports / f"{name}-validation.json", candidate)
    print(json.dumps(candidate), flush=True)
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment-v02.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/extended-42"))
    parser.add_argument("--reports", type=Path, default=Path("reports/v02"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    train, val = data(config)
    if args.output.exists():
        raise FileExistsError("Use a fresh extended run")
    args.output.mkdir(parents=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    save_json(
        args.reports / "protocol.json",
        dict(
            config=config,
            fresh_holdout=verify_freeze(config["fresh_holdout"]),
            training_pairs=len(train),
            validation_pairs=len(val),
            prior_tests_used_for_selection=False,
        ),
    )
    scores = score_checkpoint(config["initial_directory"], val, args.device)
    candidates = [
        record_candidate(
            "incumbent", config["initial_directory"], "cosine", scores, val, args.reports
        )
    ]
    for settings in config["bi_candidates"]:
        cfg = dict(config, **settings)
        cfg["model_version"] = "0.2.0-" + settings["name"]
        run = args.output / settings["name"]
        train_candidate(config["initial_directory"], train, val, run, cfg, args.device)
        directory = run / "model"
        scores = score_checkpoint(directory, val, args.device)
        candidates.append(
            record_candidate(settings["name"], directory, "cosine", scores, val, args.reports)
        )
    save_json(args.reports / "bi-development.json", dict(candidates=candidates))


if __name__ == "__main__":
    main()
