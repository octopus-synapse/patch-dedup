"""Evaluate frozen component descriptors without touching training or policy selection."""

import gc
from pathlib import Path
from patchdedup.dataset import sha256

import numpy as np

from patchdedup.cross_encoder import CrossScorer


def component_matrix(components, rows, device="cpu"):
    import torch

    columns = []
    for component in components:
        if (
            component.get("model_sha256")
            and sha256(Path(component["directory"]) / "model.safetensors")
            != component["model_sha256"]
        ):
            raise ValueError("Selected component changed after selection")
        effective = "cpu" if component.get("quantized") else device
        scorer = CrossScorer(
            component["directory"],
            effective,
            aggregation=component.get("aggregation", "min"),
            quantized=component.get("quantized", False),
        )
        columns.append(scorer.score_pairs(rows))
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.asarray(columns, dtype=float).T
