import importlib.util

import pytest

pytestmark = pytest.mark.ml
if importlib.util.find_spec("sentence_transformers") is None:
    pytest.skip("Install the ml extra", allow_module_level=True)

from patchdedup.training import train_candidate  # noqa: E402


@pytest.mark.parametrize("loss", ["cosine", "cosent"])
def test_training_updates_weights_and_reloads_bundle(tmp_path, loss):
    from sentence_transformers import SentenceTransformer, models
    from transformers import BertConfig, BertModel, BertTokenizerFast

    from patchdedup.encoder import Comparator
    from patchdedup.schemas import CompareInput

    # A tiny random transformer checks software only, not pretrained model quality.
    base = tmp_path / "tiny-random"
    base.mkdir()
    (base / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\nd\n")
    tokenizer = BertTokenizerFast(vocab_file=str(base / "vocab.txt"))
    tokenizer.save_pretrained(base)
    BertModel(
        BertConfig(
            vocab_size=9,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    ).save_pretrained(base)
    transformer = models.Transformer(str(base), max_seq_length=32)
    model = SentenceTransformer(modules=[transformer, models.Pooling(16)])
    initial = tmp_path / "initial"
    model.save_pretrained(initial, create_model_card=False)
    rows = [
        dict(text_a="a b", text_b="a b", binary_label=1),
        dict(text_a="a", text_b="c d", binary_label=0),
        dict(text_a="c d", text_b="c d", binary_label=1),
        dict(text_a="b", text_b="d", binary_label=0),
    ]
    config = dict(
        seed=42,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        gradient_accumulation=1,
        loss=loss,
        base_revision="tiny-random-test-only",
    )
    report = train_candidate(initial, rows, rows, tmp_path / "run", config, device="cpu")
    assert report["optimizer_steps"] == 2
    assert report["selected_parameter_delta_l2"] > 0
    bundle = tmp_path / "run" / "model"
    result = Comparator(bundle).compare(CompareInput(text_a="a b", text_b="a b"))
    assert result.similarity.score > 0.999
    with pytest.raises(FileExistsError):
        train_candidate(initial, rows, rows, tmp_path / "run", config, device="cpu")
