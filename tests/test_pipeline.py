import importlib.util
import json
import shutil

import pytest

pytestmark = pytest.mark.ml
if importlib.util.find_spec("sentence_transformers") is None:
    pytest.skip("Install ml extra", allow_module_level=True)


@pytest.mark.parametrize("layout", ["embedding_root", "nli_root"])
def test_local_pipeline_separates_cosine_score_and_uncertainty(tmp_path, layout):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from sentence_transformers import SentenceTransformer, models
    from patchdedup.encoder import Comparator, write_manifest
    from patchdedup.dataset import sha256
    from patchdedup.schemas import CompareInput

    initial = tmp_path / "nli"
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
    embedding = SentenceTransformer(
        modules=[models.Transformer(str(initial), max_seq_length=32), models.Pooling(16)]
    )
    bundle = tmp_path / "bundle"
    if layout == "nli_root":
        shutil.copytree(initial, bundle)
        embedding.save_pretrained(bundle / "embeddings", create_model_card=False)
        component_path = "."
    else:
        embedding.save_pretrained(bundle, create_model_card=False)
        shutil.copytree(initial, bundle / "components/0")
        component_path = "components/0"
    graph = dict(
        kind="average",
        indices=[0],
        weights=[1.0],
        calibrator={"kind": "identity"},
        conformal={"q": 1.0},
        components=[dict(path=component_path, aggregation="min")],
    )
    (bundle / "score_graph.json").write_text(json.dumps(graph))
    (bundle / "thresholds.json").write_text(
        json.dumps(dict(low=0, high=1, positive_enabled=False, negative_enabled=False))
    )
    (bundle / "patchdedup.json").write_text(
        json.dumps(
            dict(
                model_version="test",
                model_revision="sha256:" + sha256(bundle / "model.safetensors"),
                decision_policy_version="0.3.0",
                decision_source="cross_encoder",
                score_pipeline=True,
                embedding_directory="embeddings" if layout == "nli_root" else ".",
            )
        )
    )
    write_manifest(bundle)
    result = Comparator(bundle).compare(CompareInput(text_a="a b", text_b="a b"))
    assert result.schema_version == "0.3.0"
    assert result.similarity.score > 0.999
    assert 0 <= result.decision_score.score <= 1
    assert result.requires_review
    assert set(result.prediction_set) == {"redundant", "not_redundant"}
