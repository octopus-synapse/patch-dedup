"""Local-only sentence embeddings, verified bundles and conservative decisions."""

import json
from pathlib import Path

from patchdedup.dataset import save_json, sha256
from patchdedup.decision import decide
from patchdedup.schemas import CompareInput, CompareOutput, Thresholds


def verify_bundle(path: str | Path) -> dict:
    path = Path(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        if not isinstance(manifest, dict) or not manifest:
            raise ValueError("Empty manifest")
        required = {"model.safetensors", "patchdedup.json", "thresholds.json", "modules.json"}
        if not required <= manifest.keys():
            raise ValueError("Incomplete manifest")
        for name, digest in manifest.items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Invalid bundle path")
            if sha256(path / name) != digest:
                raise ValueError("Bundle digest mismatch")
        return manifest
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("MODEL_UNAVAILABLE: missing or invalid local bundle") from exc


def write_manifest(path: str | Path) -> None:
    path = Path(path)
    manifest = {
        str(p.relative_to(path)): sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != "manifest.json" and ".cache" not in p.parts
    }
    save_json(path / "manifest.json", manifest)


class SentenceEncoder:
    def __init__(self, path: str | Path, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            str(path), device=device, local_files_only=True, trust_remote_code=False
        )
        self.model.eval()
        self.max_length = self.model.max_seq_length

    def check_lengths(self, texts: list[str]) -> None:
        ids = self.model.tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]
        if any(len(tokens) > self.max_length for tokens in ids):
            raise ValueError("INPUT_TOO_LONG: text exceeds checkpoint token limit")

    def score_pairs(self, rows: list[dict], batch_size: int = 16) -> list[float]:
        import numpy as np
        import torch

        texts = sorted({r[key] for r in rows for key in ("text_a", "text_b")})
        self.check_lengths(texts)
        with torch.inference_mode():
            vectors = self.model.encode(
                texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or (norms <= 0).any():
            raise ValueError("INVALID_SCORE: non-finite or zero embedding")
        vectors = vectors / norms
        lookup = dict(zip(texts, vectors, strict=True))
        return [float(np.clip(lookup[r["text_a"]] @ lookup[r["text_b"]], -1, 1)) for r in rows]


class Comparator:
    def __init__(self, path: str | Path, device: str = "cpu"):
        path = Path(path)
        verify_bundle(path)
        try:
            self.metadata = json.loads((path / "patchdedup.json").read_text())
            self.thresholds = Thresholds(**json.loads((path / "thresholds.json").read_text()))
            self.encoder = SentenceEncoder(path, device)
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError("MODEL_UNAVAILABLE: cannot load local checkpoint") from exc

    def compare(self, request: CompareInput) -> CompareOutput:
        score = self.encoder.score_pairs([request.model_dump()])[0]
        decision = decide(score, self.thresholds)
        return CompareOutput(
            model_version=self.metadata["model_version"],
            model_revision=self.metadata["model_revision"],
            decision_policy_version=self.metadata["decision_policy_version"],
            similarity={"metric": "cosine", "score": score},
            decision=decision,
            thresholds=self.thresholds,
            requires_review=decision == "review",
            warnings=[
                "semantic_similarity_is_not_factual_equivalence",
                "experimental_synthetic_data_no_human_review",
            ],
        )
