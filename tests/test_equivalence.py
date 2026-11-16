import importlib.util

import pytest

pytestmark = pytest.mark.ml
if importlib.util.find_spec("transformers") is None:
    pytest.skip("Install ml extra", allow_module_level=True)

from patchdedup.equivalence import (  # noqa: E402
    BalancedBatches,
    classification_targets,
    equivalence_loss,
    select_candidate,
    train_equivalence,
)


def test_balanced_batches_are_reproducible_and_include_both_classes():
    import numpy as np

    labels = [1, 1] + [0] * 18
    one = list(BalancedBatches(labels, 4, seed=42))
    two = list(BalancedBatches(labels, 4, seed=42))
    assert one == two and len(one) == 5
    assert all(sum(labels[i] for i in batch) == 2 for batch in one)
    assert all(len(set(batch)) == 4 for batch in one)
    weights = np.ones(20)
    weights[4] = 100
    hard = BalancedBatches(labels, 4, seed=42, weights=weights)
    assert any(4 in batch for batch in hard)
    with pytest.raises(ValueError):
        BalancedBatches([0] * 10, 4, seed=42)


def test_three_way_labels_are_assin_relations_not_directional_nli():
    rows = [dict(binary_label=int(label == 2), original_label=label) for label in range(3)]
    assert classification_targets(rows, "assin_three_way") == [0, 1, 2]
    assert classification_targets(rows, "equivalence_binary") == [0, 0, 1]
    assert classification_targets(
        [
            dict(binary_label=0, relation="partial_overlap"),
            dict(binary_label=0, relation="contradictory"),
            dict(binary_label=1, relation="redundant"),
        ],
        "assin_three_way",
    ) == [1, 0, 2]


def test_loss_is_symmetric_and_ranking_is_added_exactly_once():
    import torch
    from patchdedup.nli import pairwise_ranking_loss

    ab = torch.tensor([[2.0], [-1.0]], requires_grad=True)
    ba = torch.tensor([[1.0], [-2.0]], requires_grad=True)
    y = torch.tensor([1, 0])
    plain = equivalence_loss(ab, ba, y, "equivalence_binary", ranking_weight=0)
    ranked = equivalence_loss(ab, ba, y, "equivalence_binary", ranking_weight=0.3)
    assert ranked.item() == pytest.approx(
        (plain + 0.3 * pairwise_ranking_loss(((ab + ba) / 2).flatten(), y)).item()
    )
    assert ranked.item() == pytest.approx(
        equivalence_loss(ba, ab, y, "equivalence_binary", ranking_weight=0.3).item()
    )
    ranked.backward()
    assert ab.grad.abs().sum() > 0 and ba.grad.abs().sum() > 0


def test_selection_keeps_incumbent_if_ap_or_f1_regresses():
    base = dict(name="incumbent", metrics=dict(average_precision=0.9, f1=0.8))
    worse = dict(name="worse", metrics=dict(average_precision=0.91, f1=0.79))
    better = dict(name="better", metrics=dict(average_precision=0.92, f1=0.81))
    assert select_candidate([base, worse])["name"] == "incumbent"
    assert select_candidate([base, worse, better])["name"] == "better"


@pytest.mark.parametrize("task", ["equivalence_binary", "assin_three_way"])
def test_real_training_resets_head_and_reloads_symmetric_scores(tmp_path, task):
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from patchdedup.cross_encoder import CrossScorer

    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\nd\n")
    BertTokenizerFast(vocab_file=str(initial / "vocab.txt")).save_pretrained(initial)
    mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
    model = BertForSequenceClassification(
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
    )
    with torch.no_grad():
        model.classifier.weight.fill_(20)
    model.save_pretrained(initial)
    rows = [
        dict(text_a="a b", text_b="a c", binary_label=1, original_label=2),
        dict(text_a="a b", text_b="d", binary_label=0, original_label=0),
        dict(text_a="c", text_b="d", binary_label=0, original_label=1),
        dict(text_a="b", text_b="c", binary_label=1, original_label=2),
    ] * 2
    config = dict(
        task=task,
        seed=42,
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        freeze_embeddings=True,
        freeze_lower_layers=0,
        balanced=True,
        ranking_weight=0.2,
        gradient_accumulation=1,
        gradient_checkpointing=True,
    )
    result = train_equivalence(initial, rows, rows, tmp_path / "run", config, device="cpu")
    assert result["selected_parameter_delta_l2"] > 0
    assert result["optimizer_steps"] == 2
    scorer = CrossScorer(tmp_path / "run/model", aggregation="mean")
    assert scorer.model.config.patchdedup_task == task
    assert scorer.model.config.num_labels == (1 if task == "equivalence_binary" else 3)
    assert scorer.model.classifier.weight.abs().max() < 1
    ab = scorer.score_pairs(rows[:2])
    ba = scorer.score_pairs([dict(r, text_a=r["text_b"], text_b=r["text_a"]) for r in rows[:2]])
    assert ab == pytest.approx(ba, abs=1e-7)
    assert all(0 <= s <= 1 for s in ab)
    with pytest.raises(FileExistsError):
        train_equivalence(initial, rows, rows, tmp_path / "run", config, device="cpu")
