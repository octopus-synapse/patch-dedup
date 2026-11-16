"""Symmetric joint pair scoring. Sigmoid outputs are not calibrated probabilities."""

import json
import time
from pathlib import Path

from patchdedup.dataset import save_json, sha256
from patchdedup.encoder import write_manifest
from patchdedup.evaluation import evaluate_scores, select_policy
from patchdedup.training import environment, freeze_encoder, seed_all


class CrossScorer:
    def __init__(self, directory, device="cpu", model=None, aggregation="min", quantized=False):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if quantized and device != "cpu":
            raise ValueError("MODEL_UNAVAILABLE: quantized scorer requires CPU")
        self.device = device
        self.aggregation = aggregation
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = (
            model
            if model is not None
            else AutoModelForSequenceClassification.from_pretrained(
                directory, local_files_only=True, use_safetensors=True
            ).to(device)
        )
        if quantized:
            import torch

            if device != "cpu":
                raise ValueError("MODEL_UNAVAILABLE: quantized scorer requires CPU")
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        self.model.eval()
        self.entailment_index = None
        if self.model.config.num_labels != 1:
            mapping = {key.casefold(): value for key, value in self.model.config.label2id.items()}
            if "entailment" not in mapping:
                raise ValueError("MODEL_UNAVAILABLE: NLI model must identify entailment label")
            self.entailment_index = mapping["entailment"]
        self.max_length = min(256, self.tokenizer.model_max_length)

    def tokenize(self, rows, reverse=False):
        a = [r["text_b" if reverse else "text_a"] for r in rows]
        b = [r["text_a" if reverse else "text_b"] for r in rows]
        tokens = self.tokenizer(a, b, padding=True, truncation=False, return_tensors="pt")
        if tokens["input_ids"].shape[1] > self.max_length:
            raise ValueError("INPUT_TOO_LONG: pair exceeds cross-encoder token limit")
        return {key: value.to(self.device) for key, value in tokens.items()}

    def score_pairs(self, rows, batch_size=8):
        import torch

        self.model.eval()
        scores = []
        with torch.inference_mode():
            for index in range(0, len(rows), batch_size):
                batch = rows[index : index + batch_size]
                ab = self.model(**self.tokenize(batch)).logits.float()
                ba = self.model(**self.tokenize(batch, reverse=True)).logits.float()
                if self.entailment_index is None:
                    values = (ab.view(-1).sigmoid() + ba.view(-1).sigmoid()) / 2
                else:
                    ab = ab.softmax(-1)[:, self.entailment_index]
                    ba = ba.softmax(-1)[:, self.entailment_index]
                    if self.aggregation == "min":
                        values = torch.minimum(ab, ba)
                    elif self.aggregation == "mean":
                        values = (ab + ba) / 2
                    elif self.aggregation == "geometric":
                        values = (ab * ba).clamp(min=0).sqrt()
                    else:
                        raise ValueError("Unknown entailment aggregation")
                scores.extend(values.cpu().tolist())
        return scores


def train_cross_encoder(initial, train, validation, output, config, device="cpu"):
    import torch
    from transformers import AutoModelForSequenceClassification

    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh cross-encoder experiment directory")
    output.mkdir(parents=True)
    seed_all(config["seed"])
    started = time.monotonic()
    model = AutoModelForSequenceClassification.from_pretrained(
        initial, num_labels=1, local_files_only=True, use_safetensors=True
    ).to(device)
    freeze_encoder(model, config)
    scorer = CrossScorer(initial, device, model)
    tracked = next(
        p
        for n, p in model.named_parameters()
        if n.endswith("attention.self.query.weight") and p.requires_grad
    )
    initial_parameter = tracked.detach().cpu().clone()
    loader = torch.utils.data.DataLoader(
        train,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: batch,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    positive = sum(r["binary_label"] for r in train)
    pos_weight = torch.tensor([(len(train) - positive) / positive], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["learning_rate"],
        weight_decay=0.01,
    )
    best = (-1.0, -1.0)
    history = []
    steps = 0
    accumulation = config.get("gradient_accumulation", 1)
    for epoch in range(config["epochs"]):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for index, batch in enumerate(loader):
            # Alternate orientation without duplicating scenarios across any split.
            features = scorer.tokenize(batch, reverse=bool((index + epoch) % 2))
            logits = model(**features).logits.view(-1)
            y = torch.tensor([r["binary_label"] for r in batch], dtype=torch.float32, device=device)
            loss = loss_fn(logits, y)
            total_loss += loss.item()
            start = (index // accumulation) * accumulation
            (loss / min(accumulation, len(loader) - start)).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                steps += 1
        scores = scorer.score_pairs(validation)
        labels = [r["binary_label"] for r in validation]
        policy = select_policy(labels, scores)
        metrics = evaluate_scores(labels, scores, policy)
        row = dict(epoch=epoch + 1, train_loss=total_loss / len(loader), validation=metrics)
        history.append(row)
        print(json.dumps(row), flush=True)
        objective = (metrics["average_precision"], metrics["f1"])
        if objective > best:
            best = objective
            selected_epoch = epoch + 1
            selected_delta = float(
                torch.linalg.vector_norm(tracked.detach().cpu() - initial_parameter)
            )
            selected_policy = policy
            model.save_pretrained(output / "model", safe_serialization=True)
            scorer.tokenizer.save_pretrained(output / "model")
    report = dict(
        config=config,
        history=history,
        selected_epoch=selected_epoch,
        selected_parameter_delta_l2=selected_delta,
        optimizer_steps=steps,
        policy=selected_policy,
        duration_seconds=time.monotonic() - started,
        training_pairs=len(train),
        validation_pairs=len(validation),
        environment=environment(),
        score="mean of sigmoid logits for AB and BA; uncalibrated",
    )
    save_json(output / "training.json", report)
    bundle = output / "model"
    save_json(bundle / "thresholds.json", selected_policy["thresholds"])
    save_json(
        bundle / "cross_encoder.json",
        dict(
            model_version="0.2.0-cross-experimental",
            model_revision="sha256:" + sha256(bundle / "model.safetensors"),
            training=report,
        ),
    )
    write_manifest(bundle)
    return report
