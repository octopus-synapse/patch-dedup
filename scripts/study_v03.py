"""Local v03 research stages. No stage here opens final test or calibration texts."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from patchdedup.cross_encoder import CrossScorer
from patchdedup.dataset import save_json, verify_freeze
from patchdedup.evaluation import evaluate_scores, select_policy
from patchdedup.experiment import load_rows
from patchdedup.nli import predict_logits, train_nli
from patchdedup.v03_data import augmented_training


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def data(config):
    verify_freeze("data/v03")
    development = [row for filename in config["development"] for row in load_rows(filename)]
    return augmented_training(), development


def register(name, directory, initial_key, config, output, reports, device, train_cache=False):
    train, development = data(config)
    scorer = CrossScorer(directory, device)
    logits = predict_logits(scorer, development)
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / f"{name}-development.npy", logits, allow_pickle=False)
    score = (
        torch.from_numpy(logits)
        .softmax(-1)[:, :, scorer.entailment_index]
        .min(dim=1)
        .values.numpy()
    )
    policy = select_policy([r["binary_label"] for r in development], score)
    candidate = dict(
        name=name,
        directory=str(directory),
        initial_key=initial_key,
        kind="nli",
        aggregation="min",
        policy=policy,
        validation=evaluate_scores([r["binary_label"] for r in development], score, policy),
        cache=str(cache / f"{name}-development.npy"),
    )
    if train_cache:
        teacher = predict_logits(scorer, train)
        np.save(cache / f"{name}-train.npy", teacher, allow_pickle=False)
    (reports / "candidates").mkdir(parents=True, exist_ok=True)
    save_json(reports / "candidates" / f"{name}.json", candidate)
    print(json.dumps(candidate), flush=True)
    del scorer
    cleanup()
    return candidate


def fit(name, initial, initial_key, cfg, config, output, reports, device, teacher=None, trial=None):
    run = output / name
    if not (run / "training.json").exists():
        train, development = data(config)
        report = train_nli(
            initial, train, development, run, cfg, device, teacher=teacher, trial=trial
        )
    else:
        report = json.loads((run / "training.json").read_text())
    candidate = register(name, run / "model", initial_key, config, output, reports, device)
    save_json(reports / f"{name}-training.json", report)
    return candidate, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["baseline", "train", "hpo"])
    parser.add_argument("--config", type=Path, default=Path("configs/experiment-v03.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v03"))
    parser.add_argument("--reports", type=Path, default=Path("reports/v03"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    if args.stage == "baseline":
        for name, settings in cfg["nli_models"].items():
            register(
                name,
                settings["directory"],
                name,
                cfg,
                args.output,
                args.reports,
                args.device,
                train_cache=True,
            )
        save_json(
            args.reports / "protocol.json",
            dict(
                config=cfg,
                data=verify_freeze("data/v03"),
                test_opened=False,
                calibration_opened=False,
            ),
        )
    elif args.stage == "train":
        default = dict(cfg["default_training"], seed=cfg["seed"])
        small = cfg["nli_models"]["nli_minilm"]["directory"]
        large = cfg["nli_models"]["nli_deberta"]["directory"]
        fit(
            "multitask_minilm",
            small,
            "nli_minilm",
            dict(default, freeze_lower_layers=3, rdrop=0.05, ema_decay=0.98),
            cfg,
            args.output,
            args.reports,
            args.device,
        )
        fit(
            "multitask_deberta",
            large,
            "nli_deberta",
            dict(default, **cfg["large_finetune"]),
            cfg,
            args.output,
            args.reports,
            args.device,
        )
        fit(
            "lora_deberta",
            large,
            "nli_deberta",
            dict(default, **cfg["lora"]),
            cfg,
            args.output,
            args.reports,
            args.device,
        )
        candidates = [
            json.loads(p.read_text()) for p in (args.reports / "candidates").glob("*.json")
        ]
        teachers = [c for c in candidates if c["initial_key"] == "nli_deberta"]
        teacher = max(teachers, key=lambda c: c["validation"]["average_precision"])
        scorer = CrossScorer(teacher["directory"], args.device)
        train, _ = data(cfg)
        logits = predict_logits(scorer, train)
        np.save(args.output / "cache/selected-teacher-train.npy", logits, allow_pickle=False)
        save_json(args.reports / "teacher.json", teacher)
        del scorer
        cleanup()
        fit(
            "distilled_minilm",
            small,
            "nli_minilm",
            dict(default, freeze_lower_layers=3, **cfg["distillation"]),
            cfg,
            args.output,
            args.reports,
            args.device,
            teacher=logits,
        )
        fit(
            "seed137_minilm",
            small,
            "nli_minilm",
            dict(
                default,
                freeze_lower_layers=3,
                seed=cfg["additional_seed"],
                rdrop=0.05,
                ema_decay=0.98,
            ),
            cfg,
            args.output,
            args.reports,
            args.device,
        )
    else:
        import optuna

        sampler = optuna.samplers.TPESampler(seed=cfg["seed"], n_startup_trials=2)
        pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=2, reduction_factor=2)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///{args.output / 'search.sqlite3'}",
            study_name="patchdedup-v03",
            load_if_exists=True,
        )

        def objective(trial):
            backbone = trial.suggest_categorical("backbone", cfg["hpo_backbones"])
            large = backbone == "nli_deberta"
            settings = dict(
                cfg["default_training"],
                seed=cfg["seed"],
                epochs=cfg["hpo_max_epochs"],
                learning_rate=trial.suggest_float("learning_rate", 3e-6, 3e-5, log=True),
                weight_decay=trial.suggest_float("weight_decay", 0.01, 0.1),
                label_smoothing=trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1]),
                rdrop=trial.suggest_categorical("rdrop", [0.0, 0.1]),
                freeze_lower_layers=trial.suggest_categorical(
                    f"freeze_lower_layers_{backbone}", [6, 9] if large else [0, 3]
                ),
                batch_size=4 if large else 8,
                gradient_accumulation=4 if large else 2,
                gradient_checkpointing=False,
            )
            candidate, report = fit(
                f"hpo_{trial.number}",
                cfg["nli_models"][backbone]["directory"],
                backbone,
                settings,
                cfg,
                args.output,
                args.reports,
                args.device,
                trial=trial,
            )
            if report["pruned"]:
                raise optuna.TrialPruned()
            return candidate["validation"]["average_precision"]

        remaining = max(0, cfg["hpo_trials"] - len(study.trials))
        if remaining:
            study.optimize(
                objective,
                n_trials=remaining,
                gc_after_trial=True,
                catch=(torch.cuda.OutOfMemoryError,),
            )
        save_json(
            args.reports / "hpo.json",
            dict(
                trials=[
                    dict(number=t.number, state=t.state.name, value=t.value, params=t.params)
                    for t in study.trials
                ],
                sampler="TPE, two random startup trials",
                pruner="Hyperband",
                best_trial=study.best_trial.number,
                best_value=study.best_value,
            ),
        )


if __name__ == "__main__":
    main()
