"""Supervised full fine-tuning and validation-only checkpoint selection."""

import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

from patchdedup.dataset import save_json, sha256
from patchdedup.encoder import SentenceEncoder, write_manifest
from patchdedup.evaluation import evaluate_scores, select_policy


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def environment():
    import torch

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        code_commit=commit.stdout.strip() if commit.returncode == 0 else "unavailable",
        source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        deterministic_warn_only=True,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        packages={
            name: importlib.metadata.version(name)
            for name in ("torch", "sentence-transformers", "transformers", "numpy", "scikit-learn")
        },
    )


def freeze_encoder(model, config):
    if config.get("freeze_embeddings"):
        for name, parameter in model.named_parameters():
            if ".embeddings." in name or name.startswith("embeddings."):
                parameter.requires_grad_(False)
    count = config.get("freeze_lower_layers", 0)
    for name, parameter in model.named_parameters():
        for layer in range(count):
            if f"encoder.layer.{layer}." in name:
                parameter.requires_grad_(False)


def train_candidate(initial, train, validation, output, config, device="cpu"):
    import torch
    from sentence_transformers import losses

    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh run directory; never overwrite experiments")
    output.mkdir(parents=True)
    seed_all(config["seed"])
    start = time.monotonic()
    encoder = SentenceEncoder(initial, device)
    model = encoder.model
    train = [row for row in train if row["binary_label"] is not None]
    encoder.check_lengths([row[key] for row in train + validation for key in ("text_a", "text_b")])
    freeze_encoder(model, config)
    loss_class = {
        "cosine": losses.CosineSimilarityLoss,
        "cosent": losses.CoSENTLoss,
        "contrastive": losses.ContrastiveLoss,
        "online_contrastive": losses.OnlineContrastiveLoss,
    }[config["loss"]]
    kwargs = {"margin": config.get("margin", 0.5)} if "contrastive" in config["loss"] else {}
    loss_fn = loss_class(model, **kwargs)
    sampler = None
    if config.get("sampling") == "balanced":
        from collections import Counter

        counts = Counter(r["binary_label"] for r in train)
        sampler = torch.utils.data.WeightedRandomSampler(
            [1 / counts[r["binary_label"]] for r in train],
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
    loader = torch.utils.data.DataLoader(
        train,
        batch_size=config["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=lambda batch: batch,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    tracked = next(
        p
        for name, p in model.named_parameters()
        if name.endswith("attention.self.query.weight") and p.requires_grad
    )
    initial_parameter = tracked.detach().cpu().clone()
    accumulation = config.get("gradient_accumulation", 1)
    history = []
    best = (-1.0, -1.0)
    selected_delta = None
    selected_policy = None
    steps = 0
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for index, rows in enumerate(loader):
            features = [model.tokenize([row[key] for row in rows]) for key in ("text_a", "text_b")]
            features = [
                {key: value.to(device) for key, value in feature.items()} for feature in features
            ]
            labels = torch.tensor(
                [row["binary_label"] for row in rows], dtype=torch.float32, device=device
            )
            loss = loss_fn(features, labels)
            total_loss += loss.item()
            window_start = (index // accumulation) * accumulation
            window_size = min(accumulation, len(loader) - window_start)
            (loss / window_size).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                steps += 1
        model.eval()
        scores = encoder.score_pairs(validation)
        targets = [r["binary_label"] for r in validation]
        policy = select_policy(targets, scores)
        metrics = evaluate_scores(targets, scores, policy)
        row = dict(epoch=epoch + 1, train_loss=total_loss / len(loader), validation=metrics)
        history.append(row)
        print(json.dumps(row), flush=True)
        objective = (metrics["average_precision"], metrics["f1"])
        if objective > best:
            best = objective
            selected_policy = policy
            selected_epoch = epoch + 1
            selected_delta = float(
                torch.linalg.vector_norm(tracked.detach().cpu() - initial_parameter)
            )
            model.save_pretrained(
                output / "model", safe_serialization=True, create_model_card=False
            )
    report = dict(
        config=config,
        training_pairs=len(train),
        validation_pairs=len(validation),
        selected_epoch=selected_epoch,
        selected_parameter_delta_l2=selected_delta,
        optimizer_steps=steps,
        history=history,
        policy=selected_policy,
        selection="validation average_precision, then binary F1; earliest tie",
        duration_seconds=time.monotonic() - start,
        environment=environment(),
    )
    save_json(output / "training.json", report)
    checkpoint = output / "model"
    save_json(checkpoint / "thresholds.json", selected_policy["thresholds"])
    save_json(
        checkpoint / "patchdedup.json",
        dict(
            model_version=config.get("model_version", "0.1.0-experimental"),
            model_revision="sha256:" + sha256(checkpoint / "model.safetensors"),
            decision_policy_version=selected_policy["decision_policy_version"],
            base_revision=config["base_revision"],
            training=report,
        ),
    )
    write_manifest(checkpoint)
    return report
