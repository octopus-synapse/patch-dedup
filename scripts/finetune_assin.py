"""Bounded ASSIN train/validation fine-tuning and one fixed external comparison."""

import argparse
import datetime as dt
import fcntl
import gc
import json
from pathlib import Path
import subprocess
import sys

from patchdedup.assin import convert_assin, remove_overlap
from patchdedup.dataset import sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/assin-finetune"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def status(stage, **extra):
    save(
        OUT / "status.json",
        dict(stage=stage, updated=dt.datetime.now(dt.timezone.utc).isoformat(), **extra),
    )


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def prepare():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from patchdedup.v03_data import augmented_training

    if (OUT / "protocol.json").exists():
        protocol = read(OUT / "protocol.json")
        for name, digest in protocol["files"].items():
            if sha256(OUT / name) != digest:
                raise ValueError("Prepared dataset changed")
        return protocol
    previous = ROOT / "artifacts/external-assin-ptbr"
    revision = read(previous / "protocol.json")["revision"]
    test = load_rows(previous / "pairs.jsonl")
    sources = {}
    source_hashes = {}
    for split in ("train", "validation"):
        file = hf_hub_download(
            "nilc-nlp/assin", f"ptbr/{split}/0000.parquet", repo_type="dataset", revision=revision
        )
        source_hashes[split] = sha256(file)
        sources[split] = convert_assin(pq.read_table(file).to_pylist(), split)
    local_dev = [
        row
        for file in ("data/v01/validation.jsonl", "data/v03/development.jsonl")
        for row in load_rows(ROOT / file)
    ]
    validation, removed_dev = remove_overlap(sources["validation"] + local_dev, test)
    train, removed_train = remove_overlap(
        sources["train"] + augmented_training(ROOT / "data"), test + validation
    )
    files = {}
    for split, rows in (("train", train), ("validation", validation), ("test", test)):
        file = OUT / f"{split}.jsonl"
        file.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        files[file.name] = sha256(file)
    protocol = dict(
        dataset="nilc-nlp/assin",
        revision=revision,
        source_sha256=source_hashes,
        files=files,
        train_pairs=len(train),
        validation_pairs=len(validation),
        test_pairs=len(test),
        removed_train_overlap=removed_train,
        removed_validation_overlap=removed_dev,
        positive="PARAPHRASE only",
        nli_direction="ASSIN uses binary supervision only",
        test_role="Previously inspected diagnostic; never used for epoch, model or threshold selection",
        selection="Maximum mixed-validation AP, then F1; threshold selected on validation only",
        previous_metrics=read(previous / "metrics.json"),
        publishes=False,
        max_candidates=2,
        max_epochs_each=3,
    )
    save(OUT / "protocol.json", protocol)
    return protocol


def worker(name):
    from patchdedup.nli import train_nli

    candidate = read(OUT / f"{name}.json")
    destination = OUT / name
    if destination.exists() and not (destination / "training.json").exists():
        destination.rename(OUT / f"{name}-interrupted-{dt.datetime.now().strftime('%Y%m%d%H%M%S')}")
    train_nli(
        candidate["initial"],
        load_rows(OUT / "train.jsonl"),
        load_rows(OUT / "validation.jsonl"),
        destination,
        candidate["training"],
        device="cuda",
    )


def run():
    import numpy as np
    import torch
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )
    from patchdedup.cross_encoder import CrossScorer
    from patchdedup.evaluation import select_policy

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "comparison.json").exists():
        status("complete", results=str(OUT / "comparison.json"))
        return
    status("preparing")
    protocol = prepare()
    config = dict(
        seed=137,
        epochs=3,
        batch_size=4,
        gradient_accumulation=4,
        learning_rate=8e-6,
        freeze_embeddings=True,
        freeze_lower_layers=6,
        warmup_ratio=0.1,
        weight_decay=0.01,
        label_smoothing=0.05,
        rdrop=0.0,
        ema_decay=0.0,
        ranking_weight=0.0,
        layer_lr_decay=0.9,
        patience=2,
        bf16=True,
        gradient_checkpointing=True,
    )
    candidates = [
        dict(
            name="continued",
            initial=str(ROOT / "artifacts/target-search/trial-0021/run/model"),
            training=dict(config, learning_rate=5e-6),
        ),
        dict(name="base-transfer", initial=str(ROOT / "artifacts/nli-deberta"), training=config),
    ]
    for candidate in candidates:
        candidate["initial_sha256"] = sha256(Path(candidate["initial"]) / "model.safetensors")
        path = OUT / f"{candidate['name']}.json"
        if path.exists() and read(path) != candidate:
            raise ValueError("Candidate protocol changed")
        save(path, candidate)
    validation = load_rows(OUT / "validation.jsonl")
    y = [r["binary_label"] for r in validation]
    columns = []
    for index, candidate in enumerate(candidates):
        name = candidate["name"]
        status("training", candidate=name, candidate_number=index + 1, candidates=2)
        if not (OUT / name / "training.json").exists():
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", name], check=True
            )
        status("validation", candidate=name)
        scorer = CrossScorer(OUT / name / "model", "cuda")
        columns.append(np.array(scorer.score_pairs(validation)))
        del scorer
        gc.collect()
        torch.cuda.empty_cache()
    # A small, predeclared mixture grid; all selection occurs on validation.
    choices = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        scores = weight * columns[0] + (1 - weight) * columns[1]
        policy = select_policy(y, scores)
        choices.append(
            dict(
                weight_continued=weight,
                average_precision=float(average_precision_score(y, scores)),
                f1=float(f1_score(y, scores >= policy["binary_threshold"])),
                policy=policy,
            )
        )
    selected = max(choices, key=lambda c: (c["average_precision"], c["f1"]))
    selected["components"] = [
        dict(
            directory=str(OUT / c["name"] / "model"),
            model_sha256=sha256(OUT / c["name"] / "model/model.safetensors"),
        )
        for c in candidates
    ]
    save(
        OUT / "selection.json",
        dict(selected=selected, validation_candidates=choices, test_used_for_selection=False),
    )
    status("testing")
    test = load_rows(OUT / "test.jsonl")
    test_columns = []
    for i, candidate in enumerate(candidates):
        cache = OUT / f"test-component-{i}.npy"
        if cache.exists():
            scores = np.load(cache)
        else:
            scorer = CrossScorer(OUT / candidate["name"] / "model", "cuda")
            scores = np.array(scorer.score_pairs(test))
            np.save(cache, scores)
            del scorer
            gc.collect()
            torch.cuda.empty_cache()
        test_columns.append(scores)
    weight = selected["weight_continued"]
    scores = weight * test_columns[0] + (1 - weight) * test_columns[1]
    targets = np.array([r["binary_label"] for r in test])
    pred = scores >= selected["policy"]["binary_threshold"]
    metrics = dict(
        average_precision=float(average_precision_score(targets, scores)),
        f1=float(f1_score(targets, pred)),
        precision=float(precision_score(targets, pred, zero_division=0)),
        recall=float(recall_score(targets, pred)),
        confusion_matrix=confusion_matrix(targets, pred).tolist(),
        support=len(test),
        positive_support=int(targets.sum()),
    )
    save(
        OUT / "predictions.json",
        [
            dict(r, score=float(s), prediction=int(p))
            for r, s, p in zip(test, scores, pred, strict=True)
        ],
    )
    save(
        OUT / "comparison.json",
        dict(
            before=protocol["previous_metrics"],
            after=metrics,
            selection=selected,
            limitation=protocol["test_role"],
            protocol_sha256=sha256(OUT / "protocol.json"),
        ),
    )
    status("complete", results=str(OUT / "comparison.json"), metrics=metrics)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        lock = ROOT / "artifacts/v03/orchestration/run.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run()
            except Exception as exc:
                status("failed", error=str(exc))
                raise
