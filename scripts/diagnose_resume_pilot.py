"""Diagnose contextual English pilot pairs with a fixed local ASSIN score graph.

This is experimental development inference, not the public pt-BR API or a benchmark.
No training, threshold selection, calibration fitting or frozen-test reads occur.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_row(pair):
    if pair["split"] != "development_pilot_only":
        raise ValueError("Only development pilot inputs are permitted")
    if any(not pair[side].get("model_text") for side in ("left", "right")):
        raise ValueError("Employment context is required")
    return {"text_a": pair["left"]["model_text"], "text_b": pair["right"]["model_text"]}


def summarize(predictions):
    scored = [p for p in predictions if p["status"] == "scored"]
    negatives = [p for p in scored if p["target"] == 0]
    inconclusive = [p for p in scored if p["target"] is None]
    fp = sum(p["binary_prediction"] == 1 for p in negatives)
    return dict(
        input_count=len(predictions),
        scored_count=len(scored),
        rejected_count=len(predictions) - len(scored),
        scored_negative_count=len(negatives),
        binary_false_positive_count=fp,
        binary_false_positive_rate=fp / len(negatives) if negatives else None,
        negative_decisions=dict(Counter(p["decision"] for p in negatives)),
        inconclusive_count=len(inconclusive),
        inconclusive_decisions=dict(Counter(p["decision"] for p in inconclusive)),
        limitations=[
            "AI annotations without independent human review",
            "English purposive sample from five publisher-declared real documents",
            "No positive labels: no general F1, AP, recall or bilingual-quality claim",
            "Repeated excerpts and shared document context: observations are dependent",
            "Thresholds inherited from Portuguese development, not validated for English",
            "Scores are uncalibrated decision scores, not cosine or probabilities",
        ],
    )


def write_review(output, predictions, pairs):
    pairs_by_id = {p["pair_id"]: p for p in pairs}
    fields = [
        "pair_id",
        "relation",
        "target",
        "status",
        "decision_score",
        "decision",
        "binary_prediction",
        "text_a_contextual",
        "text_b_contextual",
        "annotation_reason",
    ]
    with (output / "review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in predictions:
            source = pairs_by_id[p["pair_id"]]
            writer.writerow(
                dict(
                    **{key: p.get(key) for key in fields[:7]},
                    text_a_contextual=source["left"]["model_text"],
                    text_b_contextual=source["right"]["model_text"],
                    annotation_reason=source["rationale_ptbr"],
                )
            )


def checkpoint_files(directory):
    # All potential local inference inputs, not just the main weight file.
    return {
        str(p): sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and ".cache" not in p.parts
    }


def run(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from patchdedup.cross_encoder import CrossScorer
    from patchdedup.decision import decide
    from patchdedup.schemas import Thresholds
    from patchdedup.score_graph import apply_score_graph

    started = time.monotonic()
    torch.set_num_threads(4)
    pairs = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    rows = [model_row(p) for p in pairs]
    if not pairs or len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("Nonempty uniquely identified pilot pairs required")
    graph = read(args.graph)
    selection = read(args.selection)
    if selection["graph_sha256"] != sha256(args.graph):
        raise ValueError("Selected graph digest mismatch")
    policy = selection["selected"]["policy"]
    if policy["binary_threshold"] != graph["binary_threshold"]:
        raise ValueError("Selected binary threshold mismatch")
    thresholds = Thresholds(**policy["thresholds"])
    hashes = {str(p.resolve()): sha256(p) for p in [args.input, args.graph, args.selection]}
    directories = []
    lengths = []
    limits = []
    for component in graph["components"]:
        directory = Path(component["directory"]).resolve()
        if sha256(directory / "model.safetensors") != component["model_sha256"]:
            raise ValueError("Selected checkpoint digest mismatch")
        hashes.update(checkpoint_files(directory))
        directories.append(directory)
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        limit = min(256, tokenizer.model_max_length)
        counts = []
        for row in rows:
            ab = tokenizer(row["text_a"], row["text_b"], truncation=False)["input_ids"]
            ba = tokenizer(row["text_b"], row["text_a"], truncation=False)["input_ids"]
            counts.append(max(len(ab), len(ba)))
        limits.append(limit)
        lengths.append(counts)
    valid = [i for i in range(len(rows)) if all(n[i] <= k for n, k in zip(lengths, limits))]
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        input=str(args.input),
        source_revision=pairs[0]["left"]["source"]["revision"],
        model="local ASSIN refinement; not the published v03 bundle",
        graph=graph,
        policy=policy,
        input_hashes=hashes,
        device=args.device,
        torch_version=torch.__version__,
        max_pair_tokens=limits,
        input_representation="left/right.model_text including employment, role and period",
        truncation=False,
        score_semantics="uncalibrated cross-encoder ensemble; no embedding cosine computed",
        training=False,
        threshold_selection=False,
        frozen_test_reads=False,
        total_pairs=len(pairs),
        admitted_pairs=len(valid),
    )
    save(args.output / "protocol.json", protocol)
    print(f"Preflight: {len(valid)}/{len(rows)} pairs admitted; no truncation", flush=True)
    matrix = np.zeros((len(valid), len(directories)), dtype=float)
    for j, (directory, component) in enumerate(zip(directories, graph["components"], strict=True)):
        scorer = CrossScorer(
            directory,
            device=args.device,
            aggregation=component.get("aggregation", "min"),
            quantized=component.get("quantized", False),
        )
        for offset in range(0, len(valid), args.batch_size):
            indices = valid[offset : offset + args.batch_size]
            matrix[offset : offset + len(indices), j] = scorer.score_pairs(
                [rows[i] for i in indices], batch_size=args.batch_size
            )
            print(f"Component {j + 1}: {offset + len(indices)}/{len(valid)}", flush=True)
        del scorer
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    scores = apply_score_graph(graph, matrix) if valid else np.array([])
    positions = {index: pos for pos, index in enumerate(valid)}
    predictions = []
    for i, pair in enumerate(pairs):
        prediction = dict(
            pair_id=pair["pair_id"],
            scenario_group=pair["scenario_group"],
            relation=pair["relation"],
            target=pair["contextual_binary_target"],
            component_pair_tokens=[n[i] for n in lengths],
            status="scored" if i in positions else "input_too_long",
        )
        if i in positions:
            pos = positions[i]
            score = float(scores[pos])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid decision score")
            prediction.update(
                decision_score=score,
                component_scores=matrix[pos].tolist(),
                binary_prediction=int(score >= policy["binary_threshold"]),
                decision=decide(score, thresholds),
            )
        predictions.append(prediction)
    # Ensure inference did not change checkpoints, annotation input, graph or policy.
    if any(sha256(Path(name)) != digest for name, digest in hashes.items()):
        raise ValueError("An input changed during diagnosis")
    summary = summarize(predictions)
    summary["duration_seconds"] = time.monotonic() - started
    save(args.output / "summary.json", summary)
    save(args.output / "predictions.json", predictions)
    write_review(args.output, predictions, pairs)
    save(
        args.output / "manifest.json",
        {p.name: sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()},
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "artifacts/real_resume_pilot/prepared/pairs.jsonl"
    )
    parser.add_argument(
        "--graph", type=Path, default=ROOT / "artifacts/assin-refinement/score-graph.json"
    )
    parser.add_argument(
        "--selection", type=Path, default=ROOT / "artifacts/assin-refinement/selection.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/real_resume_pilot/diagnostics/local-assin-v1",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
