"""Specific equivalence heads, balanced training and train-only hard sampling."""

import math
import re
import time
from pathlib import Path

import numpy as np

from patchdedup.cross_encoder import CrossScorer
from patchdedup.dataset import save_json, sha256
from patchdedup.encoder import write_manifest
from patchdedup.evaluation import evaluate_scores, select_policy
from patchdedup.nli import pairwise_ranking_loss
from patchdedup.training import environment, freeze_encoder, seed_all


class BalancedBatches:
    """Half positive/half negative per microbatch, without duplicate indices inside a class."""

    def __init__(self, labels, batch_size, seed, weights=None):
        self.labels = np.asarray(labels)
        if batch_size < 2 or batch_size % 2 or set(self.labels) != {0, 1}:
            raise ValueError("Balanced batches need an even size and both binary classes")
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        self.weights = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=float)
        if self.weights.shape != self.labels.shape or not np.isfinite(self.weights).all():
            raise ValueError("Invalid sampling weights")
        if (self.weights <= 0).any():
            raise ValueError("Sampling weights must be positive")

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        count = self.batch_size // 2
        for _ in range(len(self)):
            batch = []
            for label in (0, 1):
                pool = np.flatnonzero(self.labels == label)
                p = self.weights[pool] / self.weights[pool].sum()
                batch.extend(rng.choice(pool, count, replace=len(pool) < count, p=p).tolist())
            rng.shuffle(batch)
            yield batch


def classification_targets(rows, task):
    if task == "equivalence_binary":
        return [r["binary_label"] for r in rows]
    if task != "assin_three_way":
        raise ValueError("Unknown equivalence task")
    targets = [
        r.get(
            "original_label",
            2 if r["binary_label"] == 1 else 1 if r.get("relation") == "partial_overlap" else 0,
        )
        for r in rows
    ]
    if not all(t in (0, 1, 2) for t in targets):
        raise ValueError("Invalid ASSIN relation label")
    return targets


def equivalence_loss(
    ab,
    ba,
    targets,
    task,
    ranking_weight=0,
    positive_weight=1.0,
    smoothing=0.0,
    consistency_weight=0.0,
):
    import torch
    from torch.nn import functional as F

    if task == "equivalence_binary":
        y = targets.float()
        weight = torch.as_tensor(positive_weight, device=ab.device)
        loss = 0.5 * (
            F.binary_cross_entropy_with_logits(ab.flatten(), y, pos_weight=weight)
            + F.binary_cross_entropy_with_logits(ba.flatten(), y, pos_weight=weight)
        )
        logits = ((ab + ba) / 2).flatten()
        pa, pb = ab.flatten().sigmoid(), ba.flatten().sigmoid()
    elif task == "assin_three_way":
        y = (targets == 2).float()
        loss = 0.5 * (
            F.cross_entropy(ab, targets, label_smoothing=smoothing)
            + F.cross_entropy(ba, targets, label_smoothing=smoothing)
        )
        la = ab[:, 2] - torch.logsumexp(ab[:, :2], dim=1)
        lb = ba[:, 2] - torch.logsumexp(ba[:, :2], dim=1)
        logits = (la + lb) / 2
        pa, pb = ab.softmax(-1)[:, 2], ba.softmax(-1)[:, 2]
    else:
        raise ValueError("Unknown equivalence task")
    if ranking_weight:
        loss = loss + ranking_weight * pairwise_ranking_loss(logits, y)
    if consistency_weight:
        loss = loss + consistency_weight * F.mse_loss(pa, pb)
    return loss


def select_candidate(candidates):
    """Keep the incumbent unless validation AP and F1 both do not regress."""
    baseline = candidates[0]["metrics"]
    eligible = [
        c
        for c in candidates
        if c["metrics"]["average_precision"] >= baseline["average_precision"]
        and c["metrics"]["f1"] >= baseline["f1"]
    ]
    return max(eligible, key=lambda c: (c["metrics"]["average_precision"], c["metrics"]["f1"]))


def train_equivalence(
    initial, train, validation, output, config, device="cpu", sampling_weights=None
):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification
    from transformers import get_cosine_schedule_with_warmup

    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh equivalence run directory")
    if set(r["binary_label"] for r in train) != {0, 1}:
        raise ValueError("Training requires both binary classes")
    seed_all(config["seed"])
    started = time.monotonic()
    task = config["task"]
    mapping = (
        {"equivalent": 0}
        if task == "equivalence_binary"
        else {"none": 0, "entailment": 1, "paraphrase": 2}
    )
    if task not in ("equivalence_binary", "assin_three_way"):
        raise ValueError("Unknown equivalence task")
    model_config = AutoConfig.from_pretrained(initial, local_files_only=True)
    model_config.num_labels = len(mapping)
    model_config.label2id = mapping
    model_config.id2label = {v: k for k, v in mapping.items()}
    model_config.patchdedup_task = task
    model = AutoModelForSequenceClassification.from_pretrained(
        initial,
        config=model_config,
        ignore_mismatched_sizes=True,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
    )
    # Reset even when the old NLI head also had three labels: label semantics changed.
    model.classifier.apply(model._init_weights)
    model.to(device)
    freeze_encoder(model, config)
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    scorer = CrossScorer(initial, device, model=model, aggregation="mean")
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    tracked = next(
        p for name, p in trainable if name.endswith(("query.weight", "query_proj.weight"))
    )
    original = tracked.detach().cpu().clone()
    optimizer_groups = []
    for name, parameter in trainable:
        layer = re.search(r"encoder\.layer\.(\d+)\.", name)
        scale = (
            config.get("layer_lr_decay", 0.95)
            ** (model.config.num_hidden_layers - int(layer[1]) - 1)
            if layer
            else 1.0
        )
        optimizer_groups.append(
            dict(
                params=[parameter],
                lr=config["learning_rate"] * scale,
                weight_decay=0
                if name.endswith("bias") or "LayerNorm" in name
                else config.get("weight_decay", 0.01),
            )
        )
    optimizer = torch.optim.AdamW(optimizer_groups)
    labels = [r["binary_label"] for r in train]
    if config.get("balanced", True):
        loader = torch.utils.data.DataLoader(
            train,
            batch_sampler=BalancedBatches(
                labels, config["batch_size"], config["seed"], sampling_weights
            ),
            collate_fn=lambda batch: batch,
        )
        positive_weight = 1.0
    else:
        loader = torch.utils.data.DataLoader(
            train,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=lambda batch: batch,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
        positive_weight = (len(labels) - sum(labels)) / sum(labels)
    accumulation = config.get("gradient_accumulation", 1)
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    total_steps = steps_per_epoch * config["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(total_steps * config.get("warmup_ratio", 0.1)),
        total_steps,
    )
    output.mkdir(parents=True)
    history, best, wait, steps = [], (-1, -1, -1), 0, 0
    selected_epoch = selected_policy = delta = None
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        loss_sum = 0.0
        for index, batch in enumerate(loader):
            targets = torch.tensor(classification_targets(batch, task), device=device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device == "cuda" and config.get("bf16", True),
            ):
                ab = model(**scorer.tokenize(batch)).logits.float()
                ba = model(**scorer.tokenize(batch, reverse=True)).logits.float()
                loss = equivalence_loss(
                    ab,
                    ba,
                    targets,
                    task,
                    config.get("ranking_weight", 0),
                    positive_weight,
                    config.get("label_smoothing", 0),
                    config.get("consistency_weight", 0),
                )
            loss_sum += loss.item()
            window = min(accumulation, len(loader) - (index // accumulation) * accumulation)
            (loss / window).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                steps += 1
        scores = scorer.score_pairs(validation)
        y = [r["binary_label"] for r in validation]
        policy = select_policy(y, scores)
        metrics = evaluate_scores(y, scores, policy)
        row = dict(epoch=epoch + 1, train_loss=loss_sum / len(loader), validation=metrics)
        history.append(row)
        print(__import__("json").dumps(row), flush=True)
        objective = (
            metrics["f1"] >= config.get("minimum_f1", 0),
            metrics["average_precision"],
            metrics["f1"],
        )
        if objective > best:
            best, wait, selected_epoch, selected_policy = objective, 0, epoch + 1, policy
            delta = float(torch.linalg.vector_norm(tracked.detach().cpu() - original))
            model.save_pretrained(output / "model", safe_serialization=True)
            scorer.tokenizer.save_pretrained(output / "model")
        else:
            wait += 1
        if wait >= config.get("patience", 2):
            break
    report = dict(
        config=config,
        history=history,
        selected_epoch=selected_epoch,
        policy=selected_policy,
        optimizer_steps=steps,
        selected_parameter_delta_l2=delta,
        head_reset=True,
        initial_sha256=sha256(Path(initial) / "model.safetensors"),
        training_pairs=len(train),
        validation_pairs=len(validation),
        duration_seconds=time.monotonic() - started,
        environment=environment(),
        trainable_parameters=sum(p.numel() for _, p in trainable),
        total_parameters=sum(p.numel() for p in model.parameters()),
        positive_weight=positive_weight,
        sampling="balanced microbatches"
        if config.get("balanced", True)
        else "random with class-weighted BCE",
        score="symmetric mean positive-class probability; uncalibrated",
        test_used=False,
    )
    bundle = output / "model"
    save_json(bundle / "thresholds.json", selected_policy["thresholds"])
    save_json(
        bundle / "cross_encoder.json",
        dict(
            model_version="0.4.0-equivalence-experimental",
            task=task,
            model_revision="sha256:" + sha256(bundle / "model.safetensors"),
            training=report,
        ),
    )
    write_manifest(bundle)
    # Completion marker last, after weights, tokenizer and manifest are complete.
    save_json(output / "training.json", report)
    return report
