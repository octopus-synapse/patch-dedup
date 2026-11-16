import importlib.util

import pytest

pytestmark = pytest.mark.ml
if importlib.util.find_spec("transformers") is None:
    pytest.skip("Install ml extra", allow_module_level=True)

from patchdedup.nli import directional_targets, train_nli  # noqa: E402
from patchdedup.cross_encoder import CrossScorer  # noqa: E402


def test_directional_labels_preserve_partial_information():
    labels = {"entailment": 0, "neutral": 1, "contradiction": 2}
    assert directional_targets("redundant", labels) == (0, 0)
    assert directional_targets("partial_overlap", labels) == (1, 0)
    assert directional_targets("contradictory", labels) == (2, 2)
    assert directional_targets("distinct", labels) == (1, 1)


def test_nli_training_and_symmetric_entailment_reload(tmp_path):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    initial = tmp_path / "tiny-nli"
    initial.mkdir()
    (initial / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\nd\n")
    BertTokenizerFast(vocab_file=str(initial / "vocab.txt")).save_pretrained(initial)
    mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
    BertForSequenceClassification(
        BertConfig(
            vocab_size=9,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=3,
            label2id=mapping,
            id2label={v: k for k, v in mapping.items()},
        )
    ).save_pretrained(initial)
    rows = [
        dict(text_a="a b", text_b="a c", binary_label=1, relation="redundant"),
        dict(text_a="a b", text_b="d", binary_label=0, relation="distinct"),
    ] * 3
    cfg = dict(
        seed=42,
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        gradient_accumulation=1,
        freeze_embeddings=True,
        gradient_checkpointing=True,
        warmup_ratio=0.1,
        label_smoothing=0.05,
        rdrop=0.1,
        ema_decay=0.9,
        bf16=False,
    )
    result = train_nli(initial, rows, rows, tmp_path / "run", cfg, device="cpu")
    assert result["optimizer_steps"] == 3
    assert result["selected_parameter_delta_l2"] > 0
    scorer = CrossScorer(tmp_path / "run" / "model")
    ab = scorer.score_pairs(rows[:2])
    ba = scorer.score_pairs([dict(r, text_a=r["text_b"], text_b=r["text_a"]) for r in rows[:2]])
    assert len(ab) == 2 and ab == pytest.approx(ba, abs=1e-7)
    assert all(0 <= value <= 1 for value in ab)


def test_quantized_backend_rejects_gpu_before_loading_weights(tmp_path):
    with pytest.raises(ValueError, match="MODEL_UNAVAILABLE"):
        CrossScorer(tmp_path / "absent", device="cuda", quantized=True)
