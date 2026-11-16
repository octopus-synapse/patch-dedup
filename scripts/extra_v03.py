"""Additional local ablations, confidence-filtered pseudo labels and efficiency."""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from study_v03 import cleanup, data, fit, register

from patchdedup.cross_encoder import CrossScorer
from patchdedup.dataset import save_json
from patchdedup.domain_adaptation import adapt_language_model
from patchdedup.evaluation import evaluate_scores, select_policy
from patchdedup.nli import predict_logits, train_nli
from patchdedup.pseudo import consensus_relation
from patchdedup.schemas import PairRecord


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["adapt", "pseudo", "quantize"])
    parser.add_argument("--config", type=Path, default=Path("configs/experiment-v03.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v03"))
    parser.add_argument("--reports", type=Path, default=Path("reports/v03"))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    train, development = data(config)
    default = dict(config["default_training"], seed=config["seed"])
    if args.stage == "adapt":
        initial = config["nli_models"]["nli_minilm"]["directory"]
        directory = args.output / "language_adaptation"
        if not (directory / "adaptation.json").exists():
            report = adapt_language_model(
                initial,
                [r[k] for r in train for k in ("text_a", "text_b")],
                directory,
                steps=config["domain_adaptation_steps"],
                device=args.device,
            )
            save_json(args.reports / "language-adaptation.json", report)
        fit(
            "domain_adapted_minilm",
            directory / "model",
            "nli_minilm",
            dict(default, freeze_lower_layers=3),
            config,
            args.output,
            args.reports,
            args.device,
        )
    elif args.stage == "pseudo":
        teacher = json.loads((args.reports / "teacher.json").read_text())
        groups = {}
        existing = set()
        for row in train:
            groups.setdefault(row["scenario_id"], {})[row["text_a_id"]] = row["text_a"]
            groups[row["scenario_id"]][row["text_b_id"]] = row["text_b"]
            existing.add(tuple(sorted((row["text_a"], row["text_b"]))))
        unlabelled = []
        for group, variants in groups.items():
            for index, (left, right) in enumerate(
                itertools.combinations_with_replacement(sorted(variants), 2)
            ):
                a, b = variants[left], variants[right]
                if tuple(sorted((a, b))) in existing:
                    continue
                unlabelled.append(
                    dict(
                        pair_id=f"{group}-pseudo-{index}",
                        scenario_id=group,
                        text_a_id=left,
                        text_b_id=right if right != left else right + "-view",
                        text_a=a,
                        text_b=b,
                    )
                )
        distributions = []
        for directory in (teacher["directory"], config["nli_models"]["nli_minilm"]["directory"]):
            scorer = CrossScorer(directory, args.device)
            distributions.append(predict_logits(scorer, unlabelled))
            del scorer
            cleanup()
        large, small = [torch.from_numpy(values).softmax(-1).numpy() for values in distributions]
        accepted = []
        review = []
        for row, a, b in zip(unlabelled, large, small, strict=True):
            decision = consensus_relation(a, b)
            if decision is None:
                middle = (a + b) / 2
                divergence = 0.5 * np.sum(
                    a * np.log((a + 1e-8) / (middle + 1e-8))
                    + b * np.log((b + 1e-8) / (middle + 1e-8))
                )
                review.append(
                    dict(
                        row,
                        priority=float(divergence + 1 - a.max(-1).mean()),
                        status="pending_human_review",
                    )
                )
                continue
            relation, reverse = decision
            if reverse:
                row = dict(
                    row,
                    text_a=row["text_b"],
                    text_b=row["text_a"],
                    text_a_id=row["text_b_id"],
                    text_b_id=row["text_a_id"],
                )
            row = dict(
                row,
                relation=relation,
                binary_label=int(relation == "redundant"),
                difficulty="confidence_filtered_consensus",
                domain="synthetic_professional",
                language="pt-BR",
                source_type="synthetic",
                review_status="unreviewed",
                annotation_version="0.3.0-pseudo",
                split="train",
            )
            accepted.append(PairRecord(**row).model_dump())
        save_json(
            args.reports / "pseudo-labels.json",
            dict(
                candidates=len(unlabelled),
                accepted=len(accepted),
                confidence=0.98,
                teacher=teacher["name"],
                agreement="NLI MiniLM directional argmax agreement",
                annotation="machine pseudo labels; no human review",
            ),
        )
        save_json(
            args.reports / "active-review-queue.json",
            dict(
                reviewed_by_human=False, examples=sorted(review, key=lambda r: -r["priority"])[:50]
            ),
        )
        pseudo_path = args.output / "pseudo-train.jsonl"
        pseudo_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted))
        if not accepted:
            return
        full = train + accepted
        scorer = CrossScorer(teacher["directory"], args.device)
        soft = predict_logits(scorer, full)
        del scorer
        cleanup()
        run = args.output / "pseudo_distilled_minilm"
        cfg = dict(
            default,
            freeze_lower_layers=3,
            distillation_weight=0.3,
            temperature=2.0,
            hard_sampling=True,
        )
        if not (run / "training.json").exists():
            report = train_nli(
                config["nli_models"]["nli_minilm"]["directory"],
                full,
                development,
                run,
                cfg,
                args.device,
                teacher=soft,
            )
            save_json(args.reports / "pseudo_distilled_minilm-training.json", report)
        register(
            "pseudo_distilled_minilm",
            run / "model",
            "nli_minilm",
            config,
            args.output,
            args.reports,
            args.device,
        )
    else:
        torch.set_num_threads(2)
        candidates = [
            json.loads(p.read_text())
            for p in (args.reports / "candidates").glob("*.json")
            if "quantized" not in p.name
        ]
        best = max(candidates, key=lambda c: c["validation"]["average_precision"])
        timings = {}
        for quantized in (False, True):
            scorer = CrossScorer(best["directory"], "cpu", quantized=quantized)
            scorer.score_pairs(development[:4])
            duration = []
            for _ in range(3):
                started = time.monotonic()
                scorer.score_pairs(development[:4])
                duration.append(time.monotonic() - started)
            timings["int8" if quantized else "float32"] = float(np.median(duration))
            if quantized:
                logits = predict_logits(scorer, development)
                name = best["name"] + "_quantized"
                cache = args.output / "cache" / f"{name}-development.npy"
                np.save(cache, logits, allow_pickle=False)
                scores = (
                    torch.from_numpy(logits)
                    .softmax(-1)[:, :, scorer.entailment_index]
                    .min(1)
                    .values.numpy()
                )
                policy = select_policy([r["binary_label"] for r in development], scores)
                candidate = dict(
                    best,
                    name=name,
                    quantized=True,
                    cache=str(cache),
                    policy=policy,
                    validation=evaluate_scores(
                        [r["binary_label"] for r in development], scores, policy
                    ),
                )
                save_json(args.reports / "candidates" / f"{name}.json", candidate)
            del scorer
            cleanup()
        save_json(
            args.reports / "quantization.json",
            dict(
                source=best["name"],
                cpu_threads=2,
                median_seconds_per_four_pairs=timings,
                precision="dynamic int8 Linear; embeddings float32",
            ),
        )


if __name__ == "__main__":
    main()
