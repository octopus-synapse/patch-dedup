"""Domain-language adaptation ablation on training texts, with an explicit new MLM head."""

import gc
import time
from pathlib import Path

from patchdedup.dataset import save_json
from patchdedup.training import freeze_encoder, seed_all


def adapt_language_model(initial, texts, output, steps=40, device="cpu"):
    import torch
    from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling

    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh language adaptation directory")
    output.mkdir(parents=True)
    seed_all(42)
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(initial, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
        initial, local_files_only=True, use_safetensors=True, torch_dtype=torch.float32
    ).to(device)
    frozen = min(3, max(0, model.config.num_hidden_layers - 1))
    freeze_encoder(model, dict(freeze_embeddings=True, freeze_lower_layers=frozen))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tracked = next(
        p
        for name, p in model.named_parameters()
        if p.requires_grad and name.endswith("query.weight")
    )
    original = tracked.detach().cpu().clone()
    encoded = tokenizer(sorted(set(texts)), truncation=False)
    examples = [
        {key: value[i] for key, value in encoded.items()} for i in range(len(encoded["input_ids"]))
    ]
    if any(len(e["input_ids"]) > 128 for e in examples):
        raise ValueError("INPUT_TOO_LONG in language adaptation corpus")
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    loader = torch.utils.data.DataLoader(
        examples,
        batch_size=min(4, len(examples)),
        shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(42),
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=2e-5, weight_decay=0.01
    )
    losses = []
    iterator = iter(loader)
    model.train()
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if not (batch["labels"] != -100).any():
            batch["labels"][0, 1] = batch["input_ids"][0, 1]
            batch["input_ids"][0, 1] = tokenizer.mask_token_id
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = model(**{key: value.to(device) for key, value in batch.items()}).loss
        if not torch.isfinite(loss):
            raise ValueError("Non-finite language modeling loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss))
    delta = float(torch.linalg.vector_norm(tracked.detach().cpu() - original))
    encoder_state = {
        name: value.detach().cpu() for name, value in model.base_model.state_dict().items()
    }
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    restored = AutoModelForSequenceClassification.from_pretrained(
        initial, local_files_only=True, use_safetensors=True, torch_dtype=torch.float32
    )
    missing, unexpected = restored.base_model.load_state_dict(encoder_state, strict=False)
    if unexpected or any(not name.startswith("pooler.") for name in missing):
        raise ValueError("Incompatible encoder state during language adaptation")
    restored.save_pretrained(output / "model", safe_serialization=True)
    tokenizer.save_pretrained(output / "model")
    report = dict(
        steps=steps,
        unique_training_texts=len(examples),
        losses=losses,
        encoder_parameter_delta_l2=delta,
        seconds=time.monotonic() - start,
        newly_initialized_mlm_head=True,
        limitation="Small synthetic corpus and new MLM head; not large-scale continued pretraining",
    )
    save_json(output / "adaptation.json", report)
    return report
