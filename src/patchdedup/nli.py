"""Bidirectional NLI transfer with multitask supervision and optional distillation."""

import gc
import json
import math
import re
import time
from pathlib import Path

from patchdedup.cross_encoder import CrossScorer
from patchdedup.dataset import save_json, sha256
from patchdedup.encoder import write_manifest
from patchdedup.evaluation import evaluate_scores, select_policy
from patchdedup.training import environment, freeze_encoder, seed_all


def directional_targets(relation, labels):
    entail, neutral, contradiction = (labels[k] for k in ("entailment", "neutral", "contradiction"))
    return {
        "redundant": (entail, entail),
        "distinct": (neutral, neutral),
        "partial_overlap": (neutral, entail),
        "contradictory": (contradiction, contradiction),
    }[relation]


def predict_logits(scorer, rows, batch_size=8):
    import numpy as np
    import torch

    scorer.model.eval()
    output = []
    with torch.inference_mode():
        for index in range(0, len(rows), batch_size):
            batch = rows[index : index + batch_size]
            ab = scorer.model(**scorer.tokenize(batch)).logits.float()
            ba = scorer.model(**scorer.tokenize(batch, reverse=True)).logits.float()
            output.append(torch.stack((ab, ba), dim=1).cpu().numpy())
    return np.concatenate(output)


def train_nli(initial, train, validation, output, config, device="cpu", teacher=None, trial=None):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh NLI experiment directory")
    output.mkdir(parents=True)
    seed_all(config["seed"])
    started = time.monotonic()
    model = AutoModelForSequenceClassification.from_pretrained(
        initial, local_files_only=True, use_safetensors=True, torch_dtype=torch.float32
    ).to(device)
    freeze_encoder(model, config)
    if config.get("lora"):
        from peft import LoraConfig, TaskType, get_peft_model

        targets = (
            ["query_proj", "value_proj"]
            if model.config.model_type == "deberta-v2"
            else ["query", "value"]
        )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=targets,
            ),
        )
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if config.get("lora"):
            model.enable_input_require_grads()
    scorer = CrossScorer(initial, device, model)
    labels = {name.casefold(): value for name, value in model.config.label2id.items()}
    entail = labels["entailment"]
    others = [i for i in range(3) if i != entail]
    tracked = next(
        p
        for name, p in model.named_parameters()
        if p.requires_grad
        and any(
            name.endswith(suffix)
            for suffix in (
                "query.weight",
                "query_proj.weight",
                "query_proj.lora_B.default.weight",
                "query.lora_B.default.weight",
            )
        )
    )
    original_parameter = tracked.detach().cpu().clone()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    groups = []
    layers = model.config.num_hidden_layers
    decay = config.get("layer_lr_decay", 1.0)
    for name, parameter in trainable:
        match = re.search(r"encoder\.layer\.(\d+)\.", name)
        multiplier = decay ** (layers - int(match[1]) - 1) if match else 1.0
        groups.append(
            dict(
                params=[parameter],
                lr=config["learning_rate"] * multiplier,
                weight_decay=0.0
                if name.endswith("bias") or "LayerNorm" in name
                else config.get("weight_decay", 0.01),
            )
        )
    optimizer = torch.optim.AdamW(groups)
    sampler = None
    if config.get("hard_sampling") and teacher is not None:
        logits = torch.from_numpy(teacher)
        soft = logits.softmax(-1)[:, :, entail].min(dim=1).values.numpy()
        weights = [
            1 + (3 * float(soft[i]) if row["binary_label"] == 0 else 1.0)
            for i, row in enumerate(train)
        ]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
    indexed = list(enumerate(train))
    loader = torch.utils.data.DataLoader(
        indexed,
        batch_size=config["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=lambda batch: batch,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    accumulation = config.get("gradient_accumulation", 1)
    total_steps = math.ceil(len(loader) / accumulation) * config["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * config.get("warmup_ratio", 0.1))),
        num_training_steps=total_steps,
    )
    pos_count = sum(row["binary_label"] for row in train)
    positive_weight = torch.tensor([(len(train) - pos_count) / pos_count], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
    ema_decay = config.get("ema_decay", 0.0)
    ema = {name: p.detach().clone() for name, p in trainable} if ema_decay else None
    best, history, wait, steps, pruned = (-1.0, -1.0), [], 0, 0, False
    selected_delta = selected_policy = selected_epoch = None
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for index, entries in enumerate(loader):
            ids, batch = zip(*entries, strict=True)
            batch = list(batch)
            y = torch.tensor([r["binary_label"] for r in batch], dtype=torch.float32, device=device)
            targets = [directional_targets(r["relation"], labels) for r in batch]
            targets_ab = torch.tensor([t[0] for t in targets], device=device)
            targets_ba = torch.tensor([t[1] for t in targets], device=device)
            features_ab, features_ba = scorer.tokenize(batch), scorer.tokenize(batch, reverse=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device == "cuda" and config.get("bf16", False),
            ):
                ab, ba = model(**features_ab).logits.float(), model(**features_ba).logits.float()
                odds_ab = ab[:, entail] - torch.logsumexp(ab[:, others], dim=1)
                odds_ba = ba[:, entail] - torch.logsumexp(ba[:, others], dim=1)
                loss = 0.5 * bce(torch.minimum(odds_ab, odds_ba), y)
                loss += 0.25 * (ce(ab, targets_ab) + ce(ba, targets_ba))
                if teacher is not None and config.get("distillation_weight", 0):
                    temperature = config.get("temperature", 2.0)
                    student = torch.stack((ab, ba), dim=1).reshape(-1, 3)
                    soft_teacher = torch.tensor(teacher[np.asarray(ids)], device=device).reshape(
                        -1, 3
                    )
                    kd = (
                        torch.nn.functional.kl_div(
                            (student / temperature).log_softmax(-1),
                            (soft_teacher / temperature).softmax(-1),
                            reduction="batchmean",
                        )
                        * temperature**2
                    )
                    weight = config["distillation_weight"]
                    loss = (1 - weight) * loss + weight * kd
                if config.get("rdrop", 0):
                    second = model(**features_ab).logits.float()
                    regularizer = 0.5 * (
                        torch.nn.functional.kl_div(
                            ab.log_softmax(-1), second.softmax(-1), reduction="batchmean"
                        )
                        + torch.nn.functional.kl_div(
                            second.log_softmax(-1), ab.softmax(-1), reduction="batchmean"
                        )
                    )
                    loss += config["rdrop"] * regularizer
            total_loss += loss.item()
            start = (index // accumulation) * accumulation
            (loss / min(accumulation, len(loader) - start)).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                steps += 1
                if ema is not None:
                    with torch.no_grad():
                        for name, parameter in trainable:
                            ema[name].lerp_(parameter.detach(), 1 - ema_decay)
        backup = None
        if ema is not None:
            backup = {name: p.detach().clone() for name, p in trainable}
            with torch.no_grad():
                for name, p in trainable:
                    p.copy_(ema[name])
        scores = scorer.score_pairs(validation)
        target = [r["binary_label"] for r in validation]
        policy = select_policy(target, scores)
        metrics = evaluate_scores(target, scores, policy)
        row = dict(epoch=epoch + 1, train_loss=total_loss / len(loader), validation=metrics)
        history.append(row)
        print(json.dumps(row), flush=True)
        objective = (metrics["average_precision"], metrics["macro_f1"])
        if objective > best:
            best, selected_epoch, selected_policy, wait = objective, epoch + 1, policy, 0
            selected_delta = float(
                torch.linalg.vector_norm(tracked.detach().cpu() - original_parameter)
            )
            destination = output / ("adapter" if config.get("lora") else "model")
            model.save_pretrained(destination, safe_serialization=True)
            scorer.tokenizer.save_pretrained(destination)
        else:
            wait += 1
        if backup is not None:
            with torch.no_grad():
                for name, p in trainable:
                    p.copy_(backup[name])
        if trial is not None:
            trial.report(metrics["average_precision"], epoch + 1)
            if trial.should_prune():
                pruned = True
                break
        if wait >= config.get("patience", 2):
            break
    report = dict(
        config=config,
        history=history,
        selected_epoch=selected_epoch,
        selected_parameter_delta_l2=selected_delta,
        optimizer_steps=steps,
        pruned=pruned,
        policy=selected_policy,
        training_pairs=len(train),
        validation_pairs=len(validation),
        trainable_parameters=sum(p.numel() for _, p in trainable),
        total_parameters=sum(p.numel() for p in model.parameters()),
        duration_seconds=time.monotonic() - started,
        environment=environment(),
        objective="multitask directional NLI CE and symmetric redundancy BCE",
        score="minimum bidirectional entailment softmax; uncalibrated",
    )
    if config.get("lora"):
        from peft import PeftModel

        del optimizer, scorer, model, trainable, ema
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        base = AutoModelForSequenceClassification.from_pretrained(
            initial, local_files_only=True, use_safetensors=True, torch_dtype=torch.float32
        )
        merged = PeftModel.from_pretrained(base, output / "adapter").merge_and_unload()
        merged.save_pretrained(output / "model", safe_serialization=True)
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(initial, local_files_only=True).save_pretrained(
            output / "model"
        )
    save_json(output / "training.json", report)
    bundle = output / "model"
    save_json(bundle / "thresholds.json", selected_policy["thresholds"])
    save_json(
        bundle / "cross_encoder.json",
        dict(
            model_version="0.3.0-nli-experimental",
            model_revision="sha256:" + sha256(bundle / "model.safetensors"),
            nli=True,
            training=report,
        ),
    )
    write_manifest(bundle)
    return report
