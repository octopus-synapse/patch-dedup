"""Bounded ASSIN head/loss ablations; select on validation, then one diagnostic test."""

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
PREVIOUS = ROOT / "artifacts/assin-finetune"
DEFAULT_OUTPUT = ROOT / "artifacts/assin-refinement"
CHILD = None


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def freeze(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f"Frozen protocol changed: {path}; use a fresh output directory")
    if not path.exists():
        save(path, value)


@contextlib.contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another local GPU controller is active; no duplicate started"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_inputs():
    from patchdedup.assin import remove_overlap
    from patchdedup.dataset import sha256

    protocol = read(PREVIOUS / "protocol.json")
    splits = {}
    for name in ("train", "validation"):
        path = PREVIOUS / f"{name}.jsonl"
        if sha256(path) != protocol["files"][path.name]:
            raise ValueError(f"Frozen {name} changed")
        splits[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    # No test labels or texts are loaded here; prior preparation already excluded shared texts.
    train, removed = remove_overlap(splits["train"], splits["validation"])
    if removed:
        raise ValueError("Training/validation overlap detected")
    validation = [r for r in splits["validation"] if r.get("source") == "nilc-nlp/assin"]
    professional = [r for r in splits["validation"] if r.get("source") != "nilc-nlp/assin"]
    return train, validation, professional


def candidate_grid(epochs):
    common = dict(
        seed=137,
        epochs=epochs,
        batch_size=4,
        gradient_accumulation=4,
        learning_rate=8e-6,
        freeze_embeddings=True,
        freeze_lower_layers=6,
        warmup_ratio=0.1,
        weight_decay=0.01,
        layer_lr_decay=0.95,
        bf16=True,
        gradient_checkpointing=True,
        patience=2,
        consistency_weight=0.05,
        task="equivalence_binary",
        balanced=True,
        ranking_weight=0.0,
        label_smoothing=0.0,
    )
    return [
        dict(name="binary-natural", training=dict(common, balanced=False), hard=False),
        dict(name="binary-balanced", training=dict(common), hard=False),
        dict(
            name="binary-ranking",
            training=dict(common, ranking_weight=0.2, learning_rate=5e-6),
            hard=False,
        ),
        dict(name="binary-hard-ranking", training=dict(common, ranking_weight=0.2), hard=True),
        dict(
            name="assin-three-way",
            training=dict(common, task="assin_three_way", label_smoothing=0.05),
            hard=False,
        ),
        dict(
            name="binary-seed42",
            training=dict(
                common, seed=42, ranking_weight=0.1, freeze_lower_layers=3, learning_rate=5e-6
            ),
            hard=True,
        ),
    ]


def worker(args):
    import numpy as np
    from patchdedup.equivalence import train_equivalence

    config = read(args.output / "configs" / f"{args.worker}.json")
    train, validation, _ = load_inputs()
    destination = args.output / "candidates" / args.worker / "run"
    if (destination / "training.json").exists():
        return
    if destination.exists():
        destination.rename(destination.with_name(f"interrupted-{time.time_ns()}"))
    weights = (
        np.load(args.output / "hard-sampling.npy", allow_pickle=False) if config["hard"] else None
    )
    train_equivalence(
        config["initial"],
        train,
        validation,
        destination,
        config["training"],
        device=args.device,
        sampling_weights=weights,
    )


def execute_worker(args, name, deadline):
    global CHILD
    logs = args.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"{name}.log").open("a", encoding="utf-8") as stream:
        CHILD = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output",
                str(args.output),
                "--device",
                args.device,
                "--worker",
                name,
            ],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            result = CHILD.wait(timeout=max(1, deadline - time.monotonic()))
        except BaseException:
            os.killpg(CHILD.pid, signal.SIGTERM)
            try:
                CHILD.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(CHILD.pid, signal.SIGKILL)
                CHILD.wait()
            raise
        finally:
            CHILD = None
    if result:
        raise RuntimeError(f"{name} failed ({result}); inspect {logs / (name + '.log')}")


def cached_scores(output, component, rows, role, device):
    import gc
    import numpy as np
    import torch
    from patchdedup.cross_encoder import CrossScorer
    from patchdedup.dataset import sha256

    directory = Path(component["directory"])
    digest = sha256(directory / "model.safetensors")
    if component.get("model_sha256", digest) != digest:
        raise ValueError("Selected weights changed")
    path = output / "cache" / f"{digest}-{role}.npy"
    if path.exists():
        result = np.load(path, allow_pickle=False)
    else:
        scorer = CrossScorer(directory, device, aggregation=component["aggregation"])
        result = np.asarray(scorer.score_pairs(rows))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, result, allow_pickle=False)
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if result.shape != (len(rows),) or not np.isfinite(result).all():
        raise ValueError("Invalid cached scores")
    return result


def component(directory, aggregation="mean"):
    from patchdedup.dataset import sha256

    return dict(
        directory=str(directory),
        aggregation=aggregation,
        model_sha256=sha256(directory / "model.safetensors"),
        quantized=False,
    )


def select(args, candidates, validation, professional):
    import itertools
    from patchdedup.dataset import sha256
    from patchdedup.equivalence import select_candidate
    from patchdedup.evaluation import evaluate_scores, select_policy

    y = [r["binary_label"] for r in validation]
    y_prof = [r["binary_label"] for r in professional]
    outcomes = []
    for i, candidate in enumerate(candidates):
        scores = cached_scores(
            args.output, candidate["component"], validation, "validation", args.device
        )
        policy = candidate.get("policy") or select_policy(y, scores)
        prof_scores = cached_scores(
            args.output, candidate["component"], professional, "professional", args.device
        )
        outcomes.append(
            dict(
                name=candidate["name"],
                metrics=evaluate_scores(y, scores, policy),
                professional_development=evaluate_scores(y_prof, prof_scores, policy),
                policy=policy,
                indices=[i],
                weights=[1.0],
            )
        )
    pool = sorted(
        range(len(outcomes)),
        key=lambda i: outcomes[i]["metrics"]["average_precision"],
        reverse=True,
    )[:3]
    if 0 not in pool:
        pool.append(0)
    # Predeclared small mixtures; every score and threshold here is development-only.
    for a, b in itertools.combinations(pool, 2):
        sa = cached_scores(
            args.output, candidates[a]["component"], validation, "validation", args.device
        )
        sb = cached_scores(
            args.output, candidates[b]["component"], validation, "validation", args.device
        )
        pa = cached_scores(
            args.output, candidates[a]["component"], professional, "professional", args.device
        )
        pb = cached_scores(
            args.output, candidates[b]["component"], professional, "professional", args.device
        )
        for w in (0.25, 0.5, 0.75):
            scores = w * sa + (1 - w) * sb
            policy = select_policy(y, scores)
            outcomes.append(
                dict(
                    name=f"{candidates[a]['name']}+{candidates[b]['name']}:{w}",
                    metrics=evaluate_scores(y, scores, policy),
                    professional_development=evaluate_scores(y_prof, w * pa + (1 - w) * pb, policy),
                    policy=policy,
                    indices=[a, b],
                    weights=[w, 1 - w],
                )
            )
    best = select_candidate(outcomes)
    components = [candidates[i]["component"] for i in best["indices"]]
    graph = dict(
        kind="average",
        indices=list(range(len(components))),
        weights=best["weights"],
        calibrator_kind="identity",
        calibrator=dict(kind="identity"),
        binary_threshold=best["policy"]["binary_threshold"],
        components=components,
    )
    save(args.output / "validation-candidates.json", outcomes)
    freeze(args.output / "score-graph.json", graph)
    selected = dict(
        selected=best,
        components=components,
        graph_sha256=sha256(args.output / "score-graph.json"),
        validation_pairs=len(validation),
        validation_positives=sum(y),
        professional_development_pairs=len(professional),
        test_used_for_selection=False,
        selection="No validation AP/F1 regression versus incumbent; then AP, F1",
    )
    freeze(args.output / "selection.json", selected)
    return selected


def evaluate_selected(args, selected):
    import numpy as np
    from patchdedup.dataset import sha256
    from patchdedup.evaluation import evaluate_scores

    protocol = read(PREVIOUS / "protocol.json")
    # Validate the frozen graph and all weights before opening the known diagnostic test.
    if sha256(args.output / "score-graph.json") != selected["graph_sha256"]:
        raise ValueError("Selected graph changed after freezing")
    for c in selected["components"]:
        if sha256(Path(c["directory"]) / "model.safetensors") != c["model_sha256"]:
            raise ValueError("Selected component changed")
    freeze(
        args.output / "test-started.json",
        dict(graph_sha256=selected["graph_sha256"], test_sha256=protocol["files"]["test.jsonl"]),
    )
    file = PREVIOUS / "test.jsonl"
    if sha256(file) != protocol["files"][file.name]:
        raise ValueError("Diagnostic test changed")
    rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
    scores = np.zeros(len(rows))
    for c, w in zip(selected["components"], selected["selected"]["weights"], strict=True):
        scores += w * cached_scores(args.output, c, rows, "diagnostic-test", args.device)
    metrics = evaluate_scores(
        [r["binary_label"] for r in rows], scores, selected["selected"]["policy"]
    )
    threshold = selected["selected"]["policy"]["binary_threshold"]
    save(
        args.output / "predictions.json",
        [
            dict(r, score=float(s), prediction=int(s >= threshold))
            for r, s in zip(rows, scores, strict=True)
        ],
    )
    before = read(PREVIOUS / "comparison.json")["after"]
    result = dict(
        before=before,
        after=metrics,
        selection=selected,
        changes={
            k: metrics[k] - before[k] for k in ("average_precision", "f1", "precision", "recall")
        },
        test_role="Repeated known ASSIN diagnostic, not a new independent test",
        independent_test=False,
        automatic_publication=False,
    )
    save(args.output / "comparison.json", result)
    lines = [
        "# Refinamento ASSIN",
        "",
        "Comparação no teste já conhecido de 2.000 pares.",
        "",
        "| Métrica | Antes | Depois |",
        "|---|---:|---:|",
    ]
    lines += [f"| {k} | {before[k]:.6f} | {metrics[k]:.6f} |" for k in result["changes"]]
    lines += [
        "",
        "Seleção apenas por validação ASSIN; a validação profissional é relatada separadamente.",
        "O teste já foi consultado em rodadas anteriores e não prova generalização independente.",
        "Nenhum serviço pago ou publicação automática. Scores não calibrados.",
    ]
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def run(args):
    import numpy as np
    from patchdedup.dataset import sha256
    from patchdedup.evaluation import evaluate_scores

    with exclusive_lock(ROOT / "artifacts/v03/orchestration/run.lock"):
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / "comparison.json").exists():
            save(
                args.output / "status.json",
                dict(stage="complete", results=str(args.output / "comparison.json")),
            )
            return
        started = time.monotonic()
        previous_status = (
            read(args.output / "status.json") if (args.output / "status.json").exists() else {}
        )
        elapsed = previous_status.get("elapsed_seconds", 0)
        deadline = started + max(0, args.max_hours * 3600 - elapsed)
        state = dict(
            stage="starting",
            pid=os.getpid(),
            max_candidates=6,
            max_hours=args.max_hours,
            test_used_for_selection=False,
        )

        def update(stage, **values):
            state.update(
                values,
                stage=stage,
                elapsed_seconds=elapsed + time.monotonic() - started,
                updated=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
            save(args.output / "status.json", state)
            print(json.dumps(state, ensure_ascii=False), flush=True)

        try:
            train, validation, professional = load_inputs()
            initial = PREVIOUS / "continued/model"
            configs = candidate_grid(args.epochs)
            protocol = dict(
                inputs={
                    name: sha256(PREVIOUS / name) for name in ("train.jsonl", "validation.jsonl")
                },
                source_protocol=read(PREVIOUS / "protocol.json"),
                initial=component(initial),
                candidates=configs,
                code_sha256={
                    name: sha256(ROOT / name)
                    for name in (
                        "scripts/refine_assin.py",
                        "src/patchdedup/equivalence.py",
                        "src/patchdedup/cross_encoder.py",
                        "src/patchdedup/nli.py",
                    )
                },
                selection="ASSIN validation AP and F1 nonregression versus incumbent, then AP/F1",
                test_role="Previously inspected diagnostic only",
                independent_test=False,
                training_pairs=len(train),
                validation_pairs=len(validation),
                professional_development_pairs=len(professional),
                paid_services=False,
                publishes=False,
            )
            freeze(args.output / "protocol.json", protocol)
            if (args.output / "selection.json").exists():
                selected = read(args.output / "selection.json")
            else:
                update("baseline_validation")
                baseline_component = component(initial, "min")
                policy = read(PREVIOUS / "selection.json")["selected"]["policy"]
                baseline_scores = cached_scores(
                    args.output, baseline_component, validation, "validation", args.device
                )
                baseline_metrics = evaluate_scores(
                    [r["binary_label"] for r in validation], baseline_scores, policy
                )
                update("training_hard_examples", incumbent_validation=baseline_metrics)
                teacher = cached_scores(
                    args.output, baseline_component, train, "training", args.device
                )
                y = np.array([r["binary_label"] for r in train])
                # Sampling uses training errors only; no test examples or pseudo-labels are added.
                np.save(
                    args.output / "hard-sampling.npy",
                    1 + 4 * np.where(y == 1, 1 - teacher, teacher),
                    allow_pickle=False,
                )
                candidates = [dict(name="incumbent", component=baseline_component, policy=policy)]
                for index, config in enumerate(configs):
                    settings = dict(config, initial=str(initial))
                    settings["training"] = dict(
                        config["training"], minimum_f1=baseline_metrics["f1"]
                    )
                    freeze(args.output / "configs" / f"{config['name']}.json", settings)
                    destination = args.output / "candidates" / config["name"] / "run"
                    if not (destination / "training.json").exists():
                        if time.monotonic() >= deadline:
                            break
                        update("training", candidate=config["name"], candidate_number=index + 1)
                        try:
                            execute_worker(args, config["name"], deadline)
                        except subprocess.TimeoutExpired:
                            update("time_budget_reached")
                            break
                    candidates.append(
                        dict(name=config["name"], component=component(destination / "model"))
                    )
                update("selecting", completed_candidates=len(candidates) - 1)
                selected = select(args, candidates, validation, professional)
            update("testing", selected=selected["selected"]["name"])
            result = evaluate_selected(args, selected)
            update(
                "complete",
                results=str(args.output / "comparison.json"),
                metrics=result["after"],
                changes=result["changes"],
                error=None,
            )
        except BaseException as exc:
            update(
                "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                error=str(exc),
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.max_hours <= 0:
        parser.error("Positive epochs and time budget required")
    args.output = args.output.resolve()
    os.chdir(ROOT)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    max_candidates=6,
                    max_hours=args.max_hours,
                    candidates=candidate_grid(args.epochs),
                    test_used_for_selection=False,
                    paid_services=False,
                    publishes=False,
                )
            )
        )
    elif args.status:
        print(
            json.dumps(read(args.output / "status.json"), indent=2, ensure_ascii=False)
            if (args.output / "status.json").exists()
            else "Ainda não iniciado."
        )
    elif args.worker:
        worker(args)
    else:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
        run(args)


if __name__ == "__main__":
    main()
