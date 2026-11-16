"""Local verified score pipelines; component weights are loaded without remote calls."""

import json
from pathlib import Path

import numpy as np

from patchdedup.calibration import prediction_sets
from patchdedup.cross_encoder import CrossScorer
from patchdedup.dataset import sha256
from patchdedup.score_graph import apply_score_graph


class ScoringPipeline:
    def __init__(self, root, device="cpu"):
        self.root = Path(root)
        self.graph = json.loads((self.root / "score_graph.json").read_text(encoding="utf-8"))
        self.revision = "sha256:" + sha256(self.root / "score_graph.json")
        self.device = device
        self.scorers = []
        for component in self.graph["components"]:
            relative = Path(component["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("MODEL_UNAVAILABLE: invalid component path")
            if component.get("quantized") and device != "cpu":
                raise ValueError("MODEL_UNAVAILABLE: selected quantized pipeline requires CPU")
            scorer = CrossScorer(
                self.root / relative,
                "cpu",
                aggregation=component.get("aggregation", "min"),
                quantized=component.get("quantized", False),
            )
            self.scorers.append(scorer)

    def score_pairs(self, rows):
        import torch

        columns = []
        for scorer in self.scorers:
            if self.device != "cpu":
                scorer.model.to(self.device)
                scorer.device = self.device
            columns.append(scorer.score_pairs(rows))
            if self.device != "cpu":
                scorer.model.to("cpu")
                scorer.device = "cpu"
                torch.cuda.empty_cache()
        return apply_score_graph(self.graph, np.asarray(columns, dtype=float).T)

    def sets(self, scores):
        return prediction_sets(scores, self.graph["conformal"]["q"])
