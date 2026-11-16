"""Cross-encoder development only; never reads held-out test texts."""

import argparse
import gc
import json
from pathlib import Path

import torch
from extend_experiment import data, record_candidate

from patchdedup.cross_encoder import CrossScorer, train_cross_encoder
from patchdedup.dataset import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiment-v02.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/extended-42"))
    parser.add_argument("--reports", type=Path, default=Path("reports/v02"))
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train, val = data(config)
    candidates = []
    for settings in config["cross_candidates"]:
        cfg = dict(config, **settings)
        name = settings["name"]
        run = args.output / name
        train_cross_encoder(config["base_directory"], train, val, run, cfg, args.device)
        scorer = CrossScorer(run / "model", args.device)
        scores = scorer.score_pairs(val)
        candidates.append(
            record_candidate(name, run / "model", "cross_encoder", scores, val, args.reports)
        )
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_json(
        args.reports / "cross-development.json",
        dict(
            candidates=candidates,
            skipped=[
                {
                    "technique": "pair feature classifier",
                    "reason": "A tool write was blocked; feature baseline was not implemented or evaluated",
                }
            ],
        ),
    )


if __name__ == "__main__":
    main()
