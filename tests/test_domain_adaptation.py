import importlib.util

import pytest

pytestmark = pytest.mark.ml
if importlib.util.find_spec("transformers") is None:
    pytest.skip("Install ml extra", allow_module_level=True)

from patchdedup.domain_adaptation import adapt_language_model  # noqa: E402


def test_domain_adaptation_updates_encoder_without_changing_label_mapping(tmp_path):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    initial = tmp_path / "tiny-nli"
    initial.mkdir()
    (initial / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\nd\n")
    BertTokenizerFast(vocab_file=str(initial / "vocab.txt")).save_pretrained(initial)
    labels = {"entailment": 0, "neutral": 1, "contradiction": 2}
    BertForSequenceClassification(
        BertConfig(
            vocab_size=9,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=3,
            label2id=labels,
            id2label={v: k for k, v in labels.items()},
        )
    ).save_pretrained(initial)
    result = adapt_language_model(
        initial, ["a b c", "b c d"], tmp_path / "adapt", steps=2, device="cpu"
    )
    assert result["steps"] == 2 and result["encoder_parameter_delta_l2"] > 0
    restored = BertForSequenceClassification.from_pretrained(
        tmp_path / "adapt/model", local_files_only=True
    )
    assert restored.config.label2id == labels
