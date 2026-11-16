"""Local, resumable, budgeted search. Never reads a calibration or final test split."""

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/target-search"
DEVELOPMENT = ["data/v01/validation.jsonl", "data/v03/development.jsonl"]
TRAINING = ["data/v01/train.jsonl", "data/refinement/train.jsonl"]


def read(file):
    return json.loads(Path(file).read_text(encoding="utf-8"))


def write(file, value):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(file)


@contextlib.contextmanager
def exclusive_lock(file):
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another controller owns the GPU/search lock") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def freeze_protocol(file, protocol):
    if file.exists():
        if read(file) != protocol:
            raise ValueError("Search protocol changed; use a fresh --output directory")
    else:
        write(file, protocol)


def plan(args):
    return dict(
        targets=dict(average_precision=args.minimum_ap, f1=args.minimum_f1),
        max_trials=args.max_trials,
        max_hours=args.max_hours,
        paid_services=False,
        reads_final_test=False,
        publishes=False,
        output=str(args.output),
        development=DEVELOPMENT,
        training=TRAINING,
        stop="Both unrounded goals on development, or time/trial limit",
        independent_test_required=True,
    )


def rows():
    from patchdedup.experiment import load_rows
    from patchdedup.v03_data import augmented_training

    train = augmented_training(ROOT / "data")
    dev = [row for file in DEVELOPMENT for row in load_rows(ROOT / file)]
    if {r["scenario_id"] for r in train} & {r["scenario_id"] for r in dev}:
        raise ValueError("Training/development scenario overlap")
    return train, dev


def register_trial(args):
    import gc
    import numpy as np
    import torch
    from patchdedup.cross_encoder import CrossScorer
    from patchdedup.nli import predict_logits, train_nli
    from patchdedup.dataset import sha256

    settings = read(args.train_config)
    train, dev = rows()
    output = args.train_config.parent
    if not (output / "run/training.json").exists():
        train_nli(
            settings["initial"],
            train,
            dev,
            output / "run",
            settings["training"],
            device=args.device,
        )
    scorer = CrossScorer(output / "run/model", args.device)
    logits = predict_logits(scorer, dev)
    np.save(output / "development.npy", logits, allow_pickle=False)
    write(
        output / "candidate.json",
        dict(
            name=output.name,
            directory=str(output / "run/model"),
            cache=str(output / "development.npy"),
            initial_key="nli_deberta",
            model_sha256=sha256(output / "run/model/model.safetensors"),
        ),
    )
    del scorer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def library(output):
    import numpy as np
    from patchdedup.dataset import sha256
    from patchdedup.experiment import load_rows

    dev = [row for name in DEVELOPMENT for row in load_rows(ROOT / name)]
    files = sorted((ROOT / "reports/v03/candidates").glob("*.json"))
    files += sorted(output.glob("trial-*/candidate.json"))
    columns, components = [], []
    for file in files:
        candidate = read(file)
        if candidate.get("quantized"):
            continue
        directory = Path(candidate["directory"])
        model_hash = sha256(directory / "model.safetensors")
        if candidate.get("model_sha256", model_hash) != model_hash:
            raise ValueError(f"Candidate weights changed: {file}")
        cache = Path(candidate["cache"])
        logits = np.load(cache, allow_pickle=False)
        if logits.shape != (len(dev), 2, 3) or not np.isfinite(logits).all():
            raise ValueError(f"Invalid development cache: {cache}")
        shifted = logits - logits.max(-1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(-1, keepdims=True)
        model_config = read(directory / "config.json")
        entailment = {k.casefold(): v for k, v in model_config["label2id"].items()}["entailment"]
        p = probabilities[:, :, entailment]
        for aggregation in ("min", "mean", "geometric"):
            columns.append(
                p.min(1)
                if aggregation == "min"
                else p.mean(1)
                if aggregation == "mean"
                else np.sqrt(p.prod(1))
            )
            components.append(
                dict(
                    name=candidate["name"] + "::" + aggregation,
                    directory=str(directory),
                    initial_key=candidate["initial_key"],
                    aggregation=aggregation,
                    quantized=False,
                    model_sha256=model_hash,
                    source_cache_sha256=sha256(cache),
                )
            )
    if not columns:
        raise ValueError("No completed local candidates available")
    return np.asarray(columns).T, components, dev


def evaluate(output, args, deadline):
    from patchdedup.target_search import metric_key, search_scores

    matrix, components, dev = library(output)
    result = search_scores(
        matrix,
        [r["binary_label"] for r in dev],
        [r["scenario_id"] for r in dev],
        args.minimum_ap,
        args.minimum_f1,
        args.ensemble_trials,
        args.seed,
        deadline,
    )
    previous = output / "search.json"
    if previous.exists() and metric_key(read(previous)["best"], args.minimum_f1) > metric_key(
        result["best"], args.minimum_f1
    ):
        return read(previous)
    graph = result.pop("graph")
    graph["components"] = [components[i] for i in graph["indices"]]
    graph["indices"] = list(range(len(graph["indices"])))
    result.update(
        development_pairs=len(dev),
        development_groups=len({r["scenario_id"] for r in dev}),
        targets=plan(args)["targets"],
        components=graph["components"],
    )
    write(output / "best-score-graph.json", graph)
    write(output / "search.json", result)
    return result


def choose_settings(trial, seed, minimum_f1):
    # Broader fine-tuning plus pairwise ranking; all initialization remains local.
    training = dict(
        seed=trial.suggest_categorical("seed", [seed, 137, 3137, 2026]),
        epochs=trial.suggest_int("epochs", 2, 5),
        batch_size=4,
        gradient_accumulation=4,
        learning_rate=trial.suggest_float("learning_rate", 1e-6, 3e-5, log=True),
        freeze_embeddings=True,
        freeze_lower_layers=trial.suggest_categorical("freeze_lower_layers", [6, 9, 10]),
        warmup_ratio=trial.suggest_categorical("warmup_ratio", [0.05, 0.1, 0.2]),
        weight_decay=trial.suggest_float("weight_decay", 0.001, 0.1, log=True),
        label_smoothing=trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1]),
        rdrop=trial.suggest_categorical("rdrop", [0.0, 0.05, 0.1]),
        ema_decay=trial.suggest_categorical("ema_decay", [0.0, 0.98]),
        ranking_weight=trial.suggest_categorical("ranking_weight", [0.0, 0.1, 0.3, 0.7]),
        layer_lr_decay=trial.suggest_categorical("layer_lr_decay", [0.9, 0.95, 1.0]),
        minimum_f1=minimum_f1,
        patience=2,
        bf16=True,
        gradient_checkpointing=True,
    )
    return dict(initial=str(ROOT / "artifacts/nli-deberta"), training=training)


def run_child(config, args, deadline):
    log = config.parent / "training.log"
    with log.open("a", encoding="utf-8") as stream:
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--train-config",
                str(config),
                "--device",
                args.device,
            ],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            child.wait(timeout=max(1, deadline - time.monotonic()))
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    if child.returncode:
        raise RuntimeError(f"Training failed ({child.returncode}); inspect {log}")


def run(args):
    import optuna
    from patchdedup.dataset import sha256

    output = args.output.resolve()
    with (
        exclusive_lock(ROOT / "artifacts/v03/orchestration/run.lock"),
        exclusive_lock(output / "run.lock"),
    ):
        protocol = dict(
            version=1,
            targets=plan(args)["targets"],
            seed=args.seed,
            ensemble_trials=args.ensemble_trials,
            input_hashes={f: sha256(ROOT / f) for f in TRAINING + DEVELOPMENT},
            reads_final_test=False,
            independent_test=False,
        )
        freeze_protocol(output / "protocol.json", protocol)
        previous = read(output / "status.json") if (output / "status.json").exists() else {}
        elapsed = previous.get("elapsed_seconds", 0.0)
        started = time.monotonic()
        deadline = started + max(0, args.max_hours * 3600 - elapsed)
        status = dict(
            status="running",
            pid=os.getpid(),
            targets=protocol["targets"],
            max_trials=args.max_trials,
            max_hours=args.max_hours,
            evaluation_split="development",
            independent_test=False,
        )

        def update(**values):
            status.update(
                values,
                elapsed_seconds=elapsed + time.monotonic() - started,
                updated=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
            write(output / "status.json", status)
            print(json.dumps(status, ensure_ascii=False), flush=True)

        try:
            update(stage="existing_candidates")
            result = evaluate(output, args, deadline)
            study = optuna.create_study(
                study_name="patchdedup-targets-v1",
                storage=f"sqlite:///{output / 'study.sqlite3'}",
                direction="maximize",
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=6),
            )
            # Resume the identical trial after interruption, preserving partial files.
            for trial in study.get_trials(deepcopy=False):
                if trial.state.name != "RUNNING":
                    continue
                folder = output / f"trial-{trial.number:04d}"
                if (folder / "candidate.json").exists():
                    study.tell(trial.number, 0.0)
                else:
                    config = folder / "config.json"
                    if config.exists():
                        saved = read(config)
                        study.enqueue_trial(trial.params, user_attrs={"resumed_from": trial.number})
                        write(folder / "interrupted.json", dict(parameters=saved, preserved=True))
                    study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            while not result["target_met"] and len(study.trials) < args.max_trials:
                if time.monotonic() >= deadline:
                    break
                trial = study.ask()
                folder = output / f"trial-{trial.number:04d}"
                settings = choose_settings(trial, args.seed, args.minimum_f1)
                config = folder / "config.json"
                write(config, settings)
                update(
                    stage="training",
                    trial=trial.number,
                    best={k: result["best"][k] for k in ("average_precision", "f1")},
                )
                try:
                    run_child(config, args, deadline)
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    write(folder / "failure.json", dict(error=str(exc)))
                    update(stage="trial_failed", error=str(exc))
                    if time.monotonic() >= deadline:
                        break
                    # Permanent errors must be visible instead of wasting the whole budget.
                    raise
                update(stage="score_search", trial=trial.number)
                result = evaluate(output, args, deadline)
                from patchdedup.target_search import search_scores

                matrix, descriptors, dev = library(output)
                indices = [
                    i for i, d in enumerate(descriptors) if d["name"].startswith(folder.name + "::")
                ]
                own = search_scores(
                    matrix[:, indices],
                    [r["binary_label"] for r in dev],
                    [r["scenario_id"] for r in dev],
                    args.minimum_ap,
                    args.minimum_f1,
                    max_ensembles=0,
                    deadline=deadline,
                )
                ap, f1 = own["best"]["average_precision"], own["best"]["f1"]
                trial.set_user_attr("average_precision", ap)
                trial.set_user_attr("f1", f1)
                study.tell(trial, ap if f1 >= args.minimum_f1 else f1 - args.minimum_f1)
            update(
                status="target_met_development" if result["target_met"] else "budget_exhausted",
                stage="finished",
                trials=len(study.trials),
                best={k: result["best"][k] for k in ("average_precision", "f1")},
                independent_test_required=True,
                error=None,
            )
        except BaseException as exc:
            update(
                status="interrupted"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "failed",
                error=str(exc),
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-ap", type=float, default=0.99)
    parser.add_argument("--minimum-f1", type=float, default=0.927)
    parser.add_argument("--max-trials", type=int, default=32)
    parser.add_argument("--max-hours", type=float, default=24)
    parser.add_argument("--ensemble-trials", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--train-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.minimum_ap <= 1 or not 0 < args.minimum_f1 <= 1:
        parser.error("Targets must lie in (0,1]")
    if args.max_trials < 0 or args.max_hours <= 0 or args.ensemble_trials < 0:
        parser.error("Invalid finite search budget")
    os.chdir(ROOT)
    if args.dry_run:
        print(json.dumps(plan(args)))
    elif args.status:
        print(
            json.dumps(read(args.output / "status.json"), indent=2, ensure_ascii=False)
            if (args.output / "status.json").exists()
            else "Busca ainda não iniciada."
        )
    elif args.train_config:
        register_trial(args)
    else:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        run(args)


if __name__ == "__main__":
    main()
