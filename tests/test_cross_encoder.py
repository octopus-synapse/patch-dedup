import importlib.util

import pytest

pytestmark = pytest.mark.ml
if importlib.util.find_spec("transformers") is None:
    pytest.skip("Install ml extra", allow_module_level=True)

from patchdedup.cross_encoder import CrossScorer, train_cross_encoder  # noqa: E402


def test_cross_encoder_training_and_symmetric_reload(tmp_path):
    from transformers import BertConfig, BertModel, BertTokenizerFast

    base = tmp_path / "tiny-random"
    base.mkdir()
    (base / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\nd\n")
    BertTokenizerFast(vocab_file=str(base / "vocab.txt")).save_pretrained(base)
    BertModel(
        BertConfig(
            vocab_size=9,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    ).save_pretrained(base)
    rows = [
        dict(text_a="a b", text_b="a c", binary_label=1),
        dict(text_a="a b", text_b="d", binary_label=0),
    ] * 3
    cfg = dict(
        seed=42,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        gradient_accumulation=1,
        base_revision="tiny-test-only",
    )
    report = train_cross_encoder(base, rows, rows, tmp_path / "run", cfg, "cpu")
    assert report["optimizer_steps"] == 3
    assert report["selected_parameter_delta_l2"] > 0
    scorer = CrossScorer(tmp_path / "run" / "model")
    ab = scorer.score_pairs(rows[:2])
    ba = scorer.score_pairs([dict(r, text_a=r["text_b"], text_b=r["text_a"]) for r in rows[:2]])
    assert ab == pytest.approx(ba, abs=1e-7)
    assert all(0 <= score <= 1 for score in ab)
    with pytest.raises(ValueError, match="INPUT_TOO_LONG"):
        scorer.score_pairs([dict(text_a="a " * 260, text_b="b")])
